"""Read only metadata columns and verify recorded sample coverage against the local benchmark."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    import pandas as pd
    cols=['sceneId','queryObjId','annotation','groundTruthObjIds','difficulty','ambiguious','split']
    paths=[args.data_root/f'train-0000{i}-of-00002.parquet' for i in range(2)]
    df=pd.concat([pd.read_parquet(path,columns=cols) for path in paths],ignore_index=True)
    checks=[]
    for difficulty in ['Easy','Medium','Hard']:
        for ambiguous in [False,True]:
            subset=difficulty.lower()+'_'+('yes' if ambiguous else 'no')+'_ambi'
            for split in ['0','1','2']:
                selected=df[(df.difficulty==difficulty)&(df.ambiguious==ambiguous)&(df.split.astype(str)==split)]
                expected={(int(r.sceneId),int(r.queryObjId)+1,r.annotation) for r in selected.itertuples()}
                path=args.result_dir/f'out_{subset}_split{split}.json'
                observed=[tuple(r[:3]) for r in json.loads(path.read_text())] if path.exists() else []
                checks.append(dict(subset=subset,split=split,expected=len(expected),observed=len(observed),
                                   missing=len(expected-set(observed)),extra=len(set(observed)-expected),
                                   duplicates=len(observed)-len(set(observed))))
    report=dict(rows=len(df),unique_scene_ids=int(df.sceneId.nunique()),
                unique_scene_target_pairs=len(df[['sceneId','queryObjId']].drop_duplicates()),checks=checks)
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else: print(text,end='')
    if any(c['missing'] or c['extra'] or c['duplicates'] for c in checks):
        raise SystemExit(1)

if __name__=='__main__': main()
