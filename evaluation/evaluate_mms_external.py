"""Zero-shot five-fold ensemble evaluation of frozen ACDC models on M&Ms."""
from __future__ import annotations

import argparse, csv, json, os, subprocess, sys
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
import torch
import yaml
from torch.amp import autocast

from evaluate_exp1_wolny_native import dice_per_class, hd95_asd
from evaluate_exp2_3dconvlstm import HALO, SPACING_DHW, retained_slice
from train_exp2_3dconvlstm import (CLASSES, PATCH_SHAPE, STRIDE,
                                   StaticUNet3DConvLSTM, UNet3DConvLSTM, starts)
from train_exp3_factorized import FactorizedUNet

MODELS = {"Exp1": ("exp1_wolny_native", None),
          "Exp2": ("exp2_3dconvlstm", UNet3DConvLSTM),
          "E2-Static": ("exp2_static_shuffle_matched", StaticUNet3DConvLSTM),
          "Exp3": ("exp3_factorized", FactorizedUNet)}
METRICS = ("RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice",
           "RV_HD95", "RV_ASD", "MYO_HD95", "MYO_ASD", "LV_HD95", "LV_ASD")
GROUPS = ("source_partition", "vendor", "centre", "pathology")


def checkpoint(args, experiment, fold):
    roots = {"Exp1": args.exp1_model_root, "Exp2": args.exp2_model_root,
             "E2-Static": args.exp2_static_model_root, "Exp3": args.exp3_model_root}
    base = roots[experiment]/f"fold_{fold}"
    return base/("checkpoints/best_checkpoint.pytorch" if experiment == "Exp1"
                 else "best_checkpoint.pytorch")


def patient_map(root):
    result = {}
    for metadata in root.glob("*/*/metadata.json"):
        patient = metadata.parent.name
        if patient in result: raise ValueError(f"duplicate patient {patient}")
        result[patient] = metadata.parent
    return result


def load_metadata(directory):
    with (directory/"metadata.json").open() as handle: return json.load(handle)


def write_csv(path, rows):
    if not rows: raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)


def load_temporal(patient, spatial_dir, temporal_root):
    partition = spatial_dir.parent.name
    temporal_dir = temporal_root/partition/patient
    xyzt = nib.load(temporal_dir/f"{patient}_cine_T30_EDaligned.nii.gz").get_fdata(dtype=np.float32)
    cine = np.ascontiguousarray(xyzt.transpose(3, 2, 1, 0))
    with (temporal_dir/"temporal_metadata.json").open() as handle:
        es_index = int(json.load(handle)["ES_normalized_index"])
    targets = []
    for phase in ("ED", "ES"):
        xyz = np.rint(nib.load(spatial_dir/f"{patient}_{phase}_gt.nii.gz").get_fdata()).astype(np.int64)
        targets.append(np.ascontiguousarray(xyz.transpose(2, 1, 0)))
    targets = np.stack(targets)
    if cine.shape != (30,16,192,192) or targets.shape != (2,16,192,192):
        raise ValueError(f"{patient}: cine={cine.shape}, target={targets.shape}")
    return cine, targets, np.asarray([[0, es_index]], dtype=np.int64)


@torch.inference_mode()
def temporal_probabilities(model, cine, phase_indices, device):
    _, depth, height, width = cine.shape
    dz, dy, dx = PATCH_SHAPE
    accum = np.zeros((2,CLASSES,depth,height,width), np.float32)
    counts = np.zeros((depth,height,width), np.float32)
    indices = torch.from_numpy(phase_indices).to(device)
    model.eval()
    for z in starts(depth,dz,STRIDE[0]):
        for y in starts(height,dy,STRIDE[1]):
            for x in starts(width,dx,STRIDE[2]):
                patch = cine[:,z:z+dz,y:y+dy,x:x+dx].copy()
                patch = (patch-float(patch.mean()))/max(float(patch.std()),1e-6)
                tensor = torch.from_numpy(patch[None,:,None]).to(device)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    probs = model(tensor,indices).softmax(dim=2)[0].float().cpu().numpy()
                pz,vz=retained_slice(z,dz,depth,HALO[0]); py,vy=retained_slice(y,dy,height,HALO[1]); px,vx=retained_slice(x,dx,width,HALO[2])
                accum[:,:,vz,vy,vx] += probs[:,:,pz,py,px]; counts[vz,vy,vx] += 1
    if np.any(counts == 0): raise RuntimeError("patch reconstruction left uncovered voxels")
    return accum/counts[None,None]


def metric_rows(patient, prediction, target, metadata):
    rows=[]
    common={"patient":patient, **{key:str(metadata[key]) for key in GROUPS}}
    for index,phase in enumerate(("ED","ES")):
        dice=dice_per_class(prediction[index],target[index],CLASSES)
        row=common|{"phase":phase,"RV_Dice":dice[1],"MYO_Dice":dice[2],
                    "LV_Dice":dice[3],"Mean_Dice":float(np.mean([dice[1],dice[2],dice[3]]))}
        for label,name in ((1,"RV"),(2,"MYO"),(3,"LV")):
            row[f"{name}_HD95"],row[f"{name}_ASD"]=hd95_asd(
                prediction[index],target[index],label,SPACING_DHW)
        rows.append(row)
    return rows


def summaries(rows):
    output=[]
    definitions=[("overall","All",rows)]
    for phase in ("ED","ES"):
        definitions.append(("phase",phase,[r for r in rows if r["phase"]==phase]))
    for group in GROUPS:
        for value in sorted({r[group] for r in rows}):
            definitions.append((group,value,[r for r in rows if r[group]==value]))
            for phase in ("ED","ES"):
                definitions.append((f"{group}_phase",f"{value}:{phase}",
                                    [r for r in rows if r[group]==value and r["phase"]==phase]))
    for family,value,selected in definitions:
        result={"group_type":family,"group_value":value,
                "n_patients":len({r["patient"] for r in selected}),"n_volumes":len(selected)}
        for metric in METRICS:
            values=np.asarray([r[metric] for r in selected],float); finite=values[np.isfinite(values)]
            result[f"{metric}_mean"]=float(finite.mean()) if len(finite) else float("nan")
            result[f"{metric}_std"]=float(finite.std(ddof=1)) if len(finite)>1 else float("nan")
        output.append(result)
    return output


def exp1_runtime_config(source, destination, h5_files, output_dir):
    with source.open() as handle: config=yaml.safe_load(handle)
    config["predictor"]["save_segmentation"]=False
    config["predictor"].pop("performance_metric",None); config["predictor"].pop("gt_internal_path",None)
    config["loaders"]["output_dir"]=str(output_dir.resolve())
    config["loaders"]["num_workers"]=0
    config["loaders"]["test"]["file_paths"]=[str(path.resolve()) for path in h5_files]
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w") as handle: yaml.safe_dump(config,handle,sort_keys=False)


def evaluate_exp1(args, patients, spatial):
    out=args.output_root/"Exp1"; probs_root=out/"fold_probabilities"
    h5_files=[]
    for patient in patients:
        partition=spatial[patient].parent.name
        h5_files.extend(args.exp1_h5_root/partition/f"{patient}_{phase}.h5" for phase in ("ED","ES"))
    for fold in range(1,6):
        fold_out=probs_root/f"fold_{fold}"; runtime=out/"runtime_configs"/f"fold_{fold}.yaml"
        source=args.exp1_model_root/"configs"/f"fold_{fold}_test_predict.yaml"
        exp1_runtime_config(source,runtime,h5_files,fold_out)
        if len(list(fold_out.glob("*_predictions.h5"))) != len(h5_files):
            env=os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
            env["PYTHONPATH"]=str((args.root/"pytorch-3dunet").resolve())
            subprocess.run([sys.executable,"-m","pytorch3dunet.predict","--config",str(runtime)],check=True,env=env)
    rows=[]
    for number,patient in enumerate(patients,1):
        partition=spatial[patient].parent.name; predictions=[]; targets=[]
        for phase in ("ED","ES"):
            name=f"{patient}_{phase}_predictions.h5"; total=None
            for fold in range(1,6):
                with h5py.File(probs_root/f"fold_{fold}"/name) as handle: p=np.asarray(handle["predictions"],np.float32)
                total=p if total is None else total+p
            predictions.append(np.argmax(total/5,axis=0).astype(np.uint8))
            with h5py.File(args.exp1_h5_root/partition/f"{patient}_{phase}.h5") as handle:
                targets.append(np.asarray(handle["label"],np.int64))
        rows.extend(metric_rows(patient,np.stack(predictions),np.stack(targets),load_metadata(spatial[patient])))
        print(f"Exp1 ensemble {number}/{len(patients)} {patient}",flush=True)
    finish_results(args,"Exp1",rows)


def evaluate_temporal(args, experiment, patients, spatial):
    out=args.output_root/experiment; probs_root=out/"fold_probabilities"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class=MODELS[experiment][1]
    for fold in range(1,6):
        fold_out=probs_root/f"fold_{fold}"; fold_out.mkdir(parents=True,exist_ok=True)
        missing=[p for p in patients if not (fold_out/f"{p}.npz").exists()]
        if not missing: continue
        state=torch.load(checkpoint(args,experiment,fold),map_location="cpu",weights_only=False)
        model=model_class().to(device); model.load_state_dict(state["model_state_dict"])
        for number,patient in enumerate(missing,1):
            cine,_,indices=load_temporal(patient,spatial[patient],args.temporal_root)
            probs=temporal_probabilities(model,cine,indices,device)
            np.savez_compressed(fold_out/f"{patient}.npz",probabilities=probs.astype(np.float16))
            print(f"{experiment} fold {fold} {number}/{len(missing)} {patient}",flush=True)
        del model; torch.cuda.empty_cache()
    rows=[]
    for number,patient in enumerate(patients,1):
        _,target,_=load_temporal(patient,spatial[patient],args.temporal_root); total=None
        for fold in range(1,6):
            with np.load(probs_root/f"fold_{fold}"/f"{patient}.npz") as archive: p=archive["probabilities"].astype(np.float32)
            total=p if total is None else total+p
        prediction=np.argmax(total/5,axis=1).astype(np.uint8)
        rows.extend(metric_rows(patient,prediction,target,load_metadata(spatial[patient])))
        print(f"{experiment} ensemble {number}/{len(patients)} {patient}",flush=True)
    finish_results(args,experiment,rows)


def finish_results(args,experiment,rows):
    out=args.output_root/experiment/"metrics"; summary=summaries(rows)
    write_csv(out/"external_volume_results.csv",rows); write_csv(out/"external_group_summary.csv",summary)
    evaluation = ("post-hoc augmentation robustness analysis" if args.augmentation_robustness_analysis else "orientation-harmonization sensitivity analysis" if args.orientation_harmonization_sensitivity else "zero-shot M&Ms external validation")
    dump={"evaluation":evaluation,"experiment":experiment,
          "post_hoc_augmentation_analysis":bool(args.augmentation_robustness_analysis),
          "post_hoc_sensitivity_analysis":bool(args.orientation_harmonization_sensitivity),
          "patients":len({r['patient'] for r in rows}),"volumes":len(rows),
          "ensemble_folds":[1,2,3,4,5],"overall":summary[0]}
    out.mkdir(parents=True,exist_ok=True); (out/"external_summary.json").write_text(json.dumps(dump,indent=2,allow_nan=True)+"\n")


def audit(args):
    spatial=patient_map(args.spatial_root); temporal={p.parent.name for p in args.temporal_root.glob("*/*/temporal_metadata.json")}
    h5={p.stem.rsplit("_",1)[0] for p in args.exp1_h5_root.glob("*/*.h5")}
    patients=sorted(spatial)
    if len(patients)!=320 or set(patients)!=temporal or set(patients)!=h5:
        raise RuntimeError(f"cohort mismatch spatial={len(patients)} temporal={len(temporal)} h5={len(h5)}")
    missing={e:[str(checkpoint(args,e,f)) for f in range(1,6) if not checkpoint(args,e,f).exists()] for e in args.experiments}
    return patients,spatial,{"patients":320,"volumes":640,"checkpoints":missing,
                            "zero_shot":not (args.orientation_harmonization_sensitivity or args.augmentation_robustness_analysis),
                            "orientation_harmonization_sensitivity":bool(args.orientation_harmonization_sensitivity),
                            "augmentation_robustness_analysis":bool(args.augmentation_robustness_analysis),"excluded":"Training/Unlabeled (25)"}


def main():
    root=Path(__file__).resolve().parent.parent; parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=root); parser.add_argument("--spatial_root",type=Path,default=root/"mms_preprocessed")
    parser.add_argument("--exp1_model_root",type=Path,default=root/"exp1_wolny_native")
    parser.add_argument("--exp2_model_root",type=Path,default=root/"exp2_3dconvlstm")
    parser.add_argument("--exp2_static_model_root",type=Path,default=root/"exp2_static_shuffle_matched")
    parser.add_argument("--exp3_model_root",type=Path,default=root/"exp3_factorized")
    parser.add_argument("--temporal_root",type=Path,default=root/"mms_temporal"); parser.add_argument("--exp1_h5_root",type=Path,default=root/"mms_exp1_h5")
    parser.add_argument("--output_root",type=Path,default=root/"mms_external_validation")
    parser.add_argument("--experiments",nargs="+",choices=MODELS,default=list(MODELS)); parser.add_argument("--gpu",default="0")
    parser.add_argument("--check_only",action="store_true"); parser.add_argument("--max_patients",type=int)
    parser.add_argument("--confirm_zero_shot_external_evaluation",action="store_true")
    parser.add_argument("--orientation_harmonization_sensitivity",action="store_true")
    parser.add_argument("--augmentation_robustness_analysis",action="store_true")
    args=parser.parse_args(); args.root=args.root.resolve()
    args.exp1_model_root=args.exp1_model_root.resolve(); args.exp2_model_root=args.exp2_model_root.resolve(); args.exp2_static_model_root=args.exp2_static_model_root.resolve(); args.exp3_model_root=args.exp3_model_root.resolve()
    patients,spatial,report=audit(args); print(json.dumps(report,indent=2))
    if args.check_only:return
    if not args.confirm_zero_shot_external_evaluation: raise SystemExit("Locked: use --confirm_zero_shot_external_evaluation")
    if any(report["checkpoints"].values()): raise RuntimeError(report["checkpoints"])
    if args.max_patients: patients=patients[:args.max_patients]
    for experiment in args.experiments:
        evaluate_exp1(args,patients,spatial) if experiment=="Exp1" else evaluate_temporal(args,experiment,patients,spatial)


if __name__=="__main__": main()
