"""Prepare Molmo marks or evaluate static mask success on FreeGraspData."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['localize', 'reasoning', 'all'], required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/new_evaluation'))
    parser.add_argument('--molmo-output', type=Path, help='Existing marked-image directory')
    parser.add_argument('--splits', nargs='+', choices=['0','1','2'], default=['0','1','2'])
    parser.add_argument('--subsets', nargs='+', default=['easy_no_ambi','easy_yes_ambi',
                        'medium_no_ambi','medium_yes_ambi','hard_no_ambi','hard_yes_ambi'])
    args = parser.parse_args()
    specs = {f'{d}_{a}_ambi': (d.title(), a=='yes')
             for d in ['easy','medium','hard'] for a in ['no','yes']}
    if any(s not in specs for s in args.subsets):
        parser.error('Unknown subset')
    files = [args.data_root/f'train-0000{i}-of-00002.parquet' for i in range(2)]
    for path in files:
        if not path.is_file():
            parser.error(f'Download dataset first: missing {path}')
    import pandas as pd
    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    df = df[df['split'].astype(str).isin(args.splits)]
    marked = args.molmo_output or args.output_root/'molmo_output'
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage in ['localize', 'all']:
        import molmo_eval
        molmo_eval.df = df
        molmo_eval.NPZ_DIR = str(args.data_root/'npz_file')
        molmo_eval.OUTPUT_DIR = str(marked)
        marked.mkdir(parents=True, exist_ok=True)
        molmo_eval.batch_process_molmo()
    if args.stage in ['reasoning', 'all']:
        from reasoning_eval import process_dataset
        reports = []
        for split in args.splits:
            for subset in args.subsets:
                difficulty, ambiguous = specs[subset]
                selected = df[(df['split'].astype(str)==split)&(df['difficulty']==difficulty)&
                              (df['ambiguious']==ambiguous)]
                if len(selected) != 50:
                    parser.error(f'{subset}/split{split}: expected 50 samples, found {len(selected)}')
                report = process_dataset(selected, args.data_root/'npz_file', marked,
                                         args.output_root/f'out_{subset}_split{split}.json')
                reports.append(dict(split=split, subset=subset, **report))
        print(json.dumps(reports, indent=2))

if __name__ == '__main__':
    main()
