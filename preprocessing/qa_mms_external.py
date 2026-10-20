"""Independent preprocessing and result QA for M&Ms external validation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, nibabel as nib, numpy as np, pandas as pd
from scipy.ndimage import zoom
from preprocess_acdc_spatial import TARGET_HW, TARGET_SPACING, center_crop_pad_xy
from analyze_exp1_wolny_errors import save_overlay

MODELS=("Exp1","Exp2","Exp3")

def source_map(root):
    result={}
    for path in root.rglob("*_sa.nii.gz"):
        patient=path.parent.name
        if path.name==f"{patient}_sa.nii.gz" and "Unlabeled" not in path.parts:
            result[patient]=path
    return result

def prediction(model,patient,phase,validation_root):
    index=0 if phase=="ED" else 1
    total=None
    if model=="Exp1":
        name=f"{patient}_{phase}_predictions.h5"
        for fold in range(1,6):
            with h5py.File(validation_root/model/"fold_probabilities"/f"fold_{fold}"/name) as f:
                value=np.asarray(f["predictions"],np.float32)
            total=value if total is None else total+value
        return np.argmax(total/5,axis=0).astype(np.int64)
    for fold in range(1,6):
        with np.load(validation_root/model/"fold_probabilities"/f"fold_{fold}"/f"{patient}.npz") as f:
            value=f["probabilities"].astype(np.float32)
        total=value if total is None else total+value
    return np.argmax(total/5,axis=1)[index].astype(np.int64)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,default=Path("data/MMS")); p.add_argument("--spatial",type=Path,default=Path("mms_preprocessed")); p.add_argument("--validation",type=Path,default=Path("mms_external_validation")); p.add_argument("--output",type=Path,default=Path("mms_external_qa")); p.add_argument("--overlays",type=int,default=5); a=p.parse_args()
    sources=source_map(a.source); records=[]
    for metadata_path in sorted(a.spatial.glob("*/*/metadata.json")):
        d=metadata_path.parent; patient=d.name; meta=json.loads(metadata_path.read_text())
        source=nib.load(sources[patient]); gt_path=sources[patient].with_name(f"{patient}_sa_gt.nii.gz"); gt=nib.load(gt_path)
        raw=np.rint(np.asanyarray(gt.dataobj)).astype(np.uint8); spacing=tuple(map(float,source.header.get_zooms()[:3])); factors=tuple(spacing[i]/TARGET_SPACING[i] for i in range(3))
        for phase in ("ED","ES"):
            frame=meta[f"{phase}_original_index_0based"]; resampled=zoom(raw[...,frame],factors,order=0,mode="nearest",prefilter=False)
            before=int(np.count_nonzero(resampled)); cropped=center_crop_pad_xy(resampled,*TARGET_HW); after=int(np.count_nonzero(cropped))
            output=np.rint(nib.load(d/f"{patient}_{phase}_gt.nii.gz").get_fdata()).astype(np.uint8)
            coords=np.argwhere(output>0); margins=[int(coords[:,axis].min()) if len(coords) else -1 for axis in range(3)]
            margins += [int(output.shape[axis]-1-coords[:,axis].max()) if len(coords) else -1 for axis in range(3)]
            records.append({"patient":patient,"phase":phase,"partition":meta["source_partition"],"vendor":meta["vendor"],"centre":meta["centre"],"pathology":meta["pathology"],"raw_shape":str(source.shape),"raw_axcodes":"".join(nib.aff2axcodes(source.affine)),"raw_spacing":str(spacing),"foreground_before_crop":before,"foreground_after_crop":after,"foreground_lost":before-after,"output_foreground":int(np.count_nonzero(output)),"bbox_min_x":margins[0],"bbox_min_y":margins[1],"bbox_min_z":margins[2],"bbox_max_margin_x":margins[3],"bbox_max_margin_y":margins[4],"bbox_max_margin_z":margins[5]})
    qa=pd.DataFrame(records); a.output.mkdir(parents=True,exist_ok=True); qa.to_csv(a.output/"preprocessing_qa.csv",index=False)
    result_checks=[]
    for model in MODELS:
        data=pd.read_csv(a.validation/model/"metrics/external_volume_results.csv")
        for metric in ("RV_Dice","MYO_Dice","LV_Dice","Mean_Dice"):
            result_checks.append({"model":model,"metric":metric,"rows":len(data),"nan":int(data[metric].isna().sum()),"zero":int((data[metric]==0).sum()),"below_0p1":int((data[metric]<.1).sum()),"min":float(data[metric].min()),"max":float(data[metric].max())})
        worst=data.nsmallest(a.overlays,"Mean_Dice")
        worst.to_csv(a.output/f"{model}_worst_cases.csv",index=False)
        for rank,row in enumerate(worst.itertuples(),1):
            directory=a.spatial/str(row.source_partition)/row.patient
            image=np.asarray(nib.load(directory/f"{row.patient}_{row.phase}.nii.gz").dataobj,np.float32).transpose(2,1,0)
            target=np.asarray(nib.load(directory/f"{row.patient}_{row.phase}_gt.nii.gz").dataobj,np.int64).transpose(2,1,0)
            pred=prediction(model,row.patient,row.phase,a.validation)
            save_overlay(image,target,pred,f"{model} #{rank} {row.patient} {row.phase} | Dice={row.Mean_Dice:.3f}",a.output/"overlays"/f"{model}_{rank:02d}_{row.patient}_{row.phase}.png")
    checks=pd.DataFrame(result_checks); checks.to_csv(a.output/"result_qa.csv",index=False)
    summary={"patients":qa.patient.nunique(),"volumes":len(qa),"foreground_lost_cases":int((qa.foreground_lost>0).sum()),"maximum_foreground_voxels_lost":int(qa.foreground_lost.max()),"orientation_counts":qa.drop_duplicates("patient").raw_axcodes.value_counts().to_dict(),"result_checks":result_checks}
    (a.output/"qa_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
