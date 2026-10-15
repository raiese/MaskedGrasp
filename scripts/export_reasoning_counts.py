"""Export counts and hashes only; do not redistribute benchmark annotations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from summarize_reasoning import count_records

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segmentation-dir', type=Path, action='append', default=[])
    p.add_argument('--target-id-json', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for root in args.segmentation_dir:
        for path in sorted(root.glob('out_*_split*.json')):
            subset, split = path.stem[4:].rsplit('_split',1)
            n,k = count_records(path)
            rows.append(dict(collection=root.name, metric='recorded_segmentation_success',
                             subset=subset, split=split, total=n, correct=k,
                             source=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if args.target_id_json:
        path = args.target_id_json
        raw = json.loads(path.read_text())['results']
        groups, seen = {}, set()
        for row in raw:
            key = (row['scene_id'],row['annotation'],str(row['split']))
            if key in seen:
                raise ValueError('Duplicate target-ID result')
            seen.add(key)
            g = (row['difficulty'].lower()+'_'+('yes' if row['ambiguious'] else 'no')+'_ambi',
                 str(row['split']))
            groups.setdefault(g,[]).append(row)
        for (subset,split), items in sorted(groups.items()):
            rows.append(dict(collection='target_id_results', metric='target_id_membership',
                             subset=subset,split=split,total=len(items),
                             correct=sum(r['is_correct'] for r in items), source=path.name,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows:
        p.error('No result records found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

if __name__ == '__main__':
    main()
