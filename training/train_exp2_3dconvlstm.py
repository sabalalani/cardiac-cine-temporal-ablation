"""Experiment 2: shared 3D U-Net encoder/decoder with 3D ConvLSTM temporal mixing."""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler


CLASSES = 4
CLASS_WEIGHTS = (0.05, 1.3, 2.0, 1.2)
PATCH_SHAPE = (8, 128, 128)
STRIDE = (4, 64, 64)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def starts(size: int, patch: int, stride: int) -> list[int]:
    values = list(range(0, max(size - patch + 1, 1), stride))
    last = size - patch
    if not values or values[-1] != last:
        values.append(last)
    return values


class TemporalPatchDataset(Dataset):
    def __init__(
        self,
        patients: list[str],
        temporal_root: Path,
        spatial_root: Path,
        training: bool,
        threshold: float = 0.01,
        augmentation_policy: str = "basic",
    ) -> None:
        self.temporal_root = temporal_root
        self.spatial_root = spatial_root
        self.training = training
        self.augmentation_policy = augmentation_policy
        self.preloaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        self.samples: list[tuple[str, tuple[int, int, int]]] = []
        dz, dy, dx = PATCH_SHAPE
        for patient in patients:
            ed = self._load_mask(patient, "ED")
            es = self._load_mask(patient, "ES")
            if ed.shape != es.shape:
                raise ValueError(f"ED/ES shape mismatch for {patient}: {ed.shape}/{es.shape}")
            union = np.logical_or(ed > 0, es > 0)
            for z in starts(ed.shape[0], dz, STRIDE[0]):
                for y in starts(ed.shape[1], dy, STRIDE[1]):
                    for x in starts(ed.shape[2], dx, STRIDE[2]):
                        fraction = float(union[z:z+dz, y:y+dy, x:x+dx].mean())
                        if fraction > threshold:
                            self.samples.append((patient, (z, y, x)))
        if not self.samples:
            raise ValueError("No temporal patches were selected")

    def _load_mask(self, patient: str, phase: str) -> np.ndarray:
        path = self.spatial_root / patient / f"{patient}_{phase}_gt.nii.gz"
        xyz = np.rint(nib.load(path).get_fdata()).astype(np.int64)
        return np.ascontiguousarray(np.transpose(xyz, (2, 1, 0)))

    @lru_cache(maxsize=2)
    def _load_patient(self, patient: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if patient in self.preloaded:
            return self.preloaded[patient]
        return self._read_patient(patient)

    def _read_patient(self, patient: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        cine_path = self.temporal_root / patient / f"{patient}_cine_T30_EDaligned.nii.gz"
        metadata_path = self.temporal_root / patient / "temporal_metadata.json"
        xyzt = nib.load(cine_path).get_fdata(dtype=np.float32)
        cine = np.ascontiguousarray(np.transpose(xyzt, (3, 2, 1, 0)))  # T,D,H,W
        ed = self._load_mask(patient, "ED")
        es = self._load_mask(patient, "ES")
        with metadata_path.open() as file:
            es_index = int(json.load(file)["ES_normalized_index"])
        if cine.shape[0] != 30 or ed.shape != cine.shape[1:] or es.shape != ed.shape:
            raise ValueError(f"Unexpected shapes for {patient}: {cine.shape}, {ed.shape}, {es.shape}")
        return cine, ed, es, es_index

    def preload_all(self) -> None:
        """Load each patient once in the parent so forked workers share RAM pages."""
        patients = sorted({patient for patient, _ in self.samples})
        total_bytes = 0
        for patient in patients:
            value = self._read_patient(patient)
            for array in value[:3]:
                array.flags.writeable = False
                total_bytes += array.nbytes
            self.preloaded[patient] = value
        print(
            f"preloaded_patients={len(patients)} "
            f"preloaded_gib={total_bytes / 1024**3:.2f}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        patient, (z, y, x) = self.samples[index]
        cine, ed, es, es_index = self._load_patient(patient)
        dz, dy, dx = PATCH_SHAPE
        cine = cine[:, z:z+dz, y:y+dy, x:x+dx].copy()
        labels = np.stack((ed[z:z+dz, y:y+dy, x:x+dx], es[z:z+dz, y:y+dy, x:x+dx])).copy()

        # Match Wolny patch standardization, applying one statistic to all frames.
        cine = (cine - float(cine.mean())) / max(float(cine.std()), 1e-6)
        if self.training:
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    cine = np.flip(cine, axis=axis).copy()
                    labels = np.flip(labels, axis=axis).copy()
            k = random.randrange(4)
            cine = np.rot90(cine, k, axes=(2, 3)).copy()
            labels = np.rot90(labels, k, axes=(2, 3)).copy()

            if self.augmentation_policy == "strong_v1":
                # One spatial transform is shared by every cine frame and label.
                angle = random.uniform(-20.0, 20.0)
                cine = rotate(cine, angle, axes=(2, 3), reshape=False,
                              order=1, mode="reflect", prefilter=False)
                labels = rotate(labels, angle, axes=(2, 3), reshape=False,
                                order=0, mode="reflect", prefilter=False)
                if random.random() < 0.20:
                    sigma = random.uniform(0.3, 0.8)
                    cine = gaussian_filter(cine, sigma=(0.0, 0.0, sigma, sigma))
                if random.random() < 0.25:
                    noise_std = random.uniform(0.0, 0.10)
                    cine = cine + np.random.normal(0.0, noise_std, cine.shape)
                if random.random() < 0.30:
                    gamma = random.uniform(0.7, 1.5)
                    cine = np.sign(cine) * np.power(np.abs(cine), gamma)
                cine = cine.astype(np.float32, copy=False)
            elif self.augmentation_policy != "basic":
                raise ValueError(f"Unknown augmentation policy: {self.augmentation_policy}")

        return {
            "cine": torch.from_numpy(cine[:, None]),  # T,C,D,H,W
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "phase_indices": torch.tensor([0, es_index], dtype=torch.long),
            "patient": patient,
        }


class PatientGroupedSampler(Sampler[int]):
    """Shuffle patients and their patches while keeping each patient's I/O contiguous.

    Temporal cines are compressed NIfTI files on network storage. Fully random
    patch order defeats the small per-worker patient cache and repeatedly
    decompresses whole 30-frame cines. This sampler preserves stochastic order
    at the patient and within-patient levels without changing the samples.
    """

    def __init__(self, dataset: TemporalPatchDataset, seed: int) -> None:
        self.groups: dict[str, list[int]] = {}
        for index, (patient, _) in enumerate(dataset.samples):
            self.groups.setdefault(patient, []).append(index)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        patients = list(self.groups)
        rng.shuffle(patients)
        for patient in patients:
            indices = self.groups[patient].copy()
            rng.shuffle(indices)
            yield from indices
        self.epoch += 1

    def __len__(self) -> int:
        return sum(map(len, self.groups.values()))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvLSTM3DCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv3d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (x.shape[0], self.hidden_channels, *x.shape[2:])
            h = x.new_zeros(shape)
            c = x.new_zeros(shape)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class BidirectionalConvLSTM3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.forward_cell = ConvLSTM3DCell(channels, hidden)
        self.backward_cell = ConvLSTM3DCell(channels, hidden)
        self.fuse = nn.Conv3d(2 * hidden, channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: B,T,C,D,H,W
        forward_states = []
        state = None
        for index in range(sequence.shape[1]):
            state = self.forward_cell(sequence[:, index], state)
            forward_states.append(state[0])
        backward_states: list[torch.Tensor | None] = [None] * sequence.shape[1]
        state = None
        for index in range(sequence.shape[1] - 1, -1, -1):
            state = self.backward_cell(sequence[:, index], state)
            backward_states[index] = state[0]
        return torch.stack(
            [self.fuse(torch.cat((forward_states[i], backward_states[i]), dim=1)) for i in range(sequence.shape[1])],
            dim=1,
        )


def gather_time(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # sequence B,T,C,D,H,W; indices B,P -> B,P,C,D,H,W
    batch = torch.arange(sequence.shape[0], device=sequence.device)[:, None]
    return sequence[batch, indices]


class UNet3DConvLSTM(nn.Module):
    def __init__(self, f_maps: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        f1, f2, f3 = f_maps
        self.enc1 = ConvBlock(1, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.bottleneck = ConvBlock(f2, f3)
        self.temporal = BidirectionalConvLSTM3D(f3)
        self.up2 = nn.ConvTranspose3d(f3, f2, 2, stride=2)
        self.dec2 = ConvBlock(f2 + f2, f2)
        self.up1 = nn.ConvTranspose3d(f2, f1, 2, stride=2)
        self.dec1 = ConvBlock(f1 + f1, f1)
        self.output = nn.Conv3d(f1, CLASSES, 1)

    def forward(self, cine: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        # cine B,T,1,D,H,W
        b, t, c, d, h, w = cine.shape
        # Encode one frame at a time. Retaining full-resolution skip maps for
        # all 30 frames is prohibitively expensive; only ED/ES need decoding.
        required_times = set(int(i) for i in phase_indices.detach().cpu().reshape(-1))
        bottlenecks = []
        skip1_at: dict[int, torch.Tensor] = {}
        skip2_at: dict[int, torch.Tensor] = {}
        for time_index in range(t):
            skip1 = self.enc1(cine[:, time_index])
            skip2 = self.enc2(F.max_pool3d(skip1, 2))
            bottlenecks.append(self.bottleneck(F.max_pool3d(skip2, 2)))
            if time_index in required_times:
                skip1_at[time_index] = skip1
                skip2_at[time_index] = skip2

        temporal = self.temporal(torch.stack(bottlenecks, dim=1))
        selected_bottleneck = gather_time(temporal, phase_indices)
        selected_skip1 = torch.stack([
            torch.stack([skip1_at[int(phase_indices[bi, pi])][bi] for pi in range(phase_indices.shape[1])])
            for bi in range(b)
        ])
        selected_skip2 = torch.stack([
            torch.stack([skip2_at[int(phase_indices[bi, pi])][bi] for pi in range(phase_indices.shape[1])])
            for bi in range(b)
        ])
        phases = phase_indices.shape[1]

        x = selected_bottleneck.reshape(b * phases, *selected_bottleneck.shape[2:])
        s2 = selected_skip2.reshape(b * phases, *selected_skip2.shape[2:])
        s1 = selected_skip1.reshape(b * phases, *selected_skip1.shape[2:])
        x = self.dec2(torch.cat((self.up2(x), s2), dim=1))
        x = self.dec1(torch.cat((self.up1(x), s1), dim=1))
        logits = self.output(x)
        return logits.reshape(b, phases, CLASSES, d, h, w)


class StaticUNet3DConvLSTM(UNet3DConvLSTM):
    """ConvLSTM control with no real inter-frame motion information.

    Each requested target phase is evaluated independently: its own image is
    repeated across all 30 time points. ED and ES therefore retain identical
    supervision and one optimizer update per source patch, while the network
    architecture and parameter names remain exactly those of Experiment 2.
    """

    @staticmethod
    def build_static_input(cine: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = cine.shape
        phases = phase_indices.shape[1]
        target_frames = gather_time(cine, phase_indices)
        return target_frames.reshape(b * phases, 1, c, d, h, w).expand(
            b * phases, t, c, d, h, w
        )

    def forward(self, cine: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        b, t, _c, d, h, w = cine.shape
        phases = phase_indices.shape[1]
        static_cines = self.build_static_input(cine, phase_indices)
        static_indices = torch.zeros((b * phases, 1), dtype=torch.long, device=cine.device)
        logits = super().forward(static_cines, static_indices)
        return logits.reshape(b, phases, CLASSES, d, h, w)


def confusion_update(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> None:
    prediction = logits.argmax(dim=2).reshape(-1)
    target = target.reshape(-1)
    valid = (target >= 0) & (target < CLASSES)
    bins = CLASSES * target[valid] + prediction[valid]
    confusion += torch.bincount(bins, minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES).cpu()


def mean_iou(confusion: torch.Tensor) -> tuple[float, list[float]]:
    scores = []
    for cls in range(1, CLASSES):
        tp = confusion[cls, cls].item()
        denominator = confusion[cls].sum().item() + confusion[:, cls].sum().item() - tp
        scores.append(tp / denominator if denominator else float("nan"))
    return float(np.nanmean(scores)), scores


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    max_batches: int | None = None,
) -> tuple[float, float, list[float]]:
    training = optimizer is not None
    model.train(training)
    losses = []
    confusion = torch.zeros((CLASSES, CLASSES), dtype=torch.int64)
    for batch_number, batch in enumerate(loader, start=1):
        cine = batch["cine"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        indices = batch["phase_indices"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(cine, indices)
            loss = criterion(logits.flatten(0, 1), labels.flatten(0, 1))
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach()))
        confusion_update(confusion, logits.detach(), labels)
        if max_batches is not None and batch_number >= max_batches:
            break
    score, classes = mean_iou(confusion)
    return float(np.mean(losses)), score, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--split_json", type=Path, default=Path("configs/acdc_5fold_splits_seed42.json"))
    parser.add_argument("--temporal_root", type=Path, default=Path("acdc_temporal/training"))
    parser.add_argument("--spatial_root", type=Path, default=Path("acdc_preprocessed/training"))
    parser.add_argument("--output_root", type=Path, default=Path("exp2_3dconvlstm"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--preload_cines", action="store_true",
                        help="Preload fold cines in parent RAM before forking workers.")
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--max_val_batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from fold last_checkpoint.pytorch")
    parser.add_argument("--augmentation_policy", choices=("basic", "strong_v1"), default="basic")
    parser.add_argument("--fixed_final_training", action="store_true",
                        help="Disable validation-driven LR changes; use last checkpoint at fixed epoch.")
    parser.add_argument("--static_input", action="store_true",
                        help="Repeat each ED/ES target image for all 30 frames (E2-Static ablation).")
    parser.add_argument(
        "--train_order",
        choices=("patient_grouped", "shuffled"),
        default="patient_grouped",
        help=("Training sample order. 'shuffled' reproduces the checkpoint-era "
              "E2-Full DataLoader(shuffle=True) behavior."),
    )
    parser.add_argument(
        "--worker_lifecycle",
        choices=("persistent", "per_epoch"),
        default="persistent",
        help=("Whether DataLoader workers persist across epochs. Checkpoint-era "
              "E2-Full used the PyTorch default 'per_epoch' behavior."),
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with args.split_json.open() as file:
        fold = json.load(file)["folds"][f"fold_{args.fold}"]
    train_set = TemporalPatchDataset(fold["train"], args.temporal_root, args.spatial_root, training=True, augmentation_policy=args.augmentation_policy)
    val_set = TemporalPatchDataset(fold["val"], args.temporal_root, args.spatial_root, training=False)
    if args.preload_cines:
        train_set.preload_all()
        val_set.preload_all()
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0 and args.worker_lifecycle == "persistent",
    }
    if args.train_order == "patient_grouped":
        train_sampler = PatientGroupedSampler(train_set, args.seed)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=train_sampler, **loader_options
        )
    else:
        # This is the ordering used by the original checkpoint-era E2-Full runs.
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **loader_options
        )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_options)

    model_class = StaticUNet3DConvLSTM if args.static_input else UNet3DConvLSTM
    model = model_class().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=8)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    output = args.output_root / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -float("inf")
    start_epoch = 0
    if args.resume:
        checkpoint_path = output / "last_checkpoint.pytorch"
        history_path = output / "history.json"
        if not checkpoint_path.exists() or not history_path.exists():
            raise FileNotFoundError(f"Cannot resume without {checkpoint_path} and {history_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with history_path.open() as file:
            history = json.load(file)
        start_epoch = int(state["epoch"])
        if len(history) != start_epoch or int(history[-1]["epoch"]) != start_epoch:
            raise RuntimeError(f"Checkpoint/history mismatch: epoch={start_epoch}, rows={len(history)}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        best = float(state["best_eval_score"])
        print(f"Resuming fold {args.fold} after epoch {start_epoch}; best_val_miou={best:.6f}", flush=True)
    max_epochs = 1 if args.smoke else args.epochs
    if start_epoch >= max_epochs:
        raise RuntimeError(f"Fold already reached {start_epoch} epochs (target={max_epochs})")
    print(f"device={device} parameters={parameters:,} train_patches={len(train_set)} val_patches={len(val_set)}")
    for epoch in range(start_epoch, max_epochs):
        start = time.perf_counter()
        train_limit = args.max_train_batches
        val_limit = args.max_val_batches
        if args.smoke:
            train_limit = train_limit or 2
            val_limit = val_limit or 2
        train_loss, train_score, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, train_limit
        )
        with torch.no_grad():
            val_loss, val_score, class_scores = run_epoch(
                model, val_loader, criterion, device, None, scaler, val_limit
            )
        if not args.fixed_final_training:
            scheduler.step(val_score)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_miou": train_score,
            "val_loss": val_loss,
            "val_miou": val_score,
            "RV_IoU": class_scores[0],
            "MYO_IoU": class_scores[1],
            "LV_IoU": class_scores[2],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_eval_score": max(best, val_score),
            "parameters": parameters,
            "config": vars(args),
        }
        torch.save(checkpoint, output / "last_checkpoint.pytorch")
        if val_score > best:
            best = val_score
            checkpoint["best_eval_score"] = best
            torch.save(checkpoint, output / "best_checkpoint.pytorch")
        with (output / "history.json").open("w") as file:
            json.dump(history, file, indent=2)
        if optimizer.param_groups[0]["lr"] < 1e-6:
            print("Learning rate below 1e-6; stopping.")
            break


if __name__ == "__main__":
    main()
