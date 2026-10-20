"""Experiment 3: factorized (3+1)D spatiotemporal U-Net."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

try:
    from training.train_exp2_3dconvlstm import (
        CLASSES,
        CLASS_WEIGHTS,
        PatientGroupedSampler,
        TemporalPatchDataset,
        gather_time,
        run_epoch,
        seed_everything,
    )
except ModuleNotFoundError:  # direct execution from training/
    from train_exp2_3dconvlstm import (
        CLASSES,
        CLASS_WEIGHTS,
        PatientGroupedSampler,
        TemporalPatchDataset,
        gather_time,
        run_epoch,
        seed_everything,
    )


def framewise(module: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
    """Apply a 3D spatial module independently to each temporal frame."""
    b, c, t, d, h, w = sequence.shape
    frames = sequence.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
    frames = module(frames)
    return frames.reshape(b, t, *frames.shape[1:]).permute(0, 2, 1, 3, 4, 5).contiguous()


class FactorizedConv(nn.Module):
    """A 3D spatial convolution followed by a 1D temporal convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.spatial = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.spatial_norm = nn.GroupNorm(groups, out_channels)
        self.temporal = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.temporal_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = framewise(self.spatial, x)
        x = F.relu(self.spatial_norm(x), inplace=True)
        b, c, t, d, h, w = x.shape
        # Apply the same per-voxel temporal Conv1d as a batched Conv3d with a
        # (time, 1, 1) kernel. This keeps the parameter/checkpoint shapes and
        # mathematical operation unchanged while avoiding millions of tiny
        # Conv1d sequences and their expensive flattened layout.
        temporal_volumes = x.permute(0, 3, 1, 2, 4, 5).reshape(b * d, c, t, h, w)
        temporal_volumes = F.conv3d(
            temporal_volumes,
            self.temporal.weight.unsqueeze(-1).unsqueeze(-1),
            self.temporal.bias,
            stride=(self.temporal.stride[0], 1, 1),
            padding=(self.temporal.padding[0], 0, 0),
            dilation=(self.temporal.dilation[0], 1, 1),
            groups=self.temporal.groups,
        )
        x = temporal_volumes.reshape(b, d, c, t, h, w).permute(0, 2, 3, 1, 4, 5).contiguous()
        return F.relu(self.temporal_norm(x), inplace=True)


class FactorizedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first = FactorizedConv(in_channels, out_channels)
        self.second = FactorizedConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


class FactorizedUNet(nn.Module):
    def __init__(self, f_maps: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        f1, f2, f3 = f_maps
        self.enc1 = FactorizedBlock(1, f1)
        self.enc2 = FactorizedBlock(f1, f2)
        self.bottleneck = FactorizedBlock(f2, f3)
        self.up2 = nn.ConvTranspose3d(f3, f2, 2, stride=2)
        self.dec2 = FactorizedBlock(f2 + f2, f2)
        self.up1 = nn.ConvTranspose3d(f2, f1, 2, stride=2)
        self.dec1 = FactorizedBlock(f1 + f1, f1)
        self.output = nn.Conv3d(f1, CLASSES, 1)

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        return framewise(nn.MaxPool3d(2), x)

    def forward(self, cine: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        # Dataset layout B,T,C,D,H,W -> joint feature layout B,C,T,D,H,W.
        x = cine.permute(0, 2, 1, 3, 4, 5).contiguous()
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        x = self.bottleneck(self.pool(skip2))
        x = self.dec2(torch.cat((framewise(self.up2, x), skip2), dim=1))
        x = self.dec1(torch.cat((framewise(self.up1, x), skip1), dim=1))
        logits = framewise(self.output, x).permute(0, 2, 1, 3, 4, 5).contiguous()
        return gather_time(logits, phase_indices)


class StaticFactorizedUNet(FactorizedUNet):
    """Experiment 3 with target-specific repeated frames and no cardiac motion."""

    @staticmethod
    def build_static_input(cine: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = cine.shape
        phases = phase_indices.shape[1]
        targets = gather_time(cine, phase_indices)
        return targets.reshape(b * phases, 1, c, d, h, w).expand(
            b * phases, t, c, d, h, w
        )

    def forward(self, cine: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        b, _t, _c, d, h, w = cine.shape
        phases = phase_indices.shape[1]
        static_cines = self.build_static_input(cine, phase_indices)
        static_indices = torch.zeros((b * phases, 1), dtype=torch.long, device=cine.device)
        logits = super().forward(static_cines, static_indices)
        return logits.reshape(b, phases, CLASSES, d, h, w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--split_json", type=Path, default=Path("configs/acdc_5fold_splits_seed42.json"))
    parser.add_argument("--temporal_root", type=Path, default=Path("acdc_temporal/training"))
    parser.add_argument("--spatial_root", type=Path, default=Path("acdc_preprocessed/training"))
    parser.add_argument("--output_root", type=Path, default=Path("exp3_factorized"))
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
                        help="Repeat each ED/ES target image for all 30 frames (E3-Static ablation).")
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
    train_sampler = PatientGroupedSampler(train_set, args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=train_sampler, **loader_options)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            **loader_options)

    model_class = StaticFactorizedUNet if args.static_input else FactorizedUNet
    model = model_class().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=8
    )
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
        started = time.perf_counter()
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
            "seconds": time.perf_counter() - started,
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
