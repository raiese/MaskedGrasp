"""Recompute split means and both SD conventions, without model dependencies."""
import argparse
import csv
import json
import statistics
from pathlib import Path

SUBSETS = ('easy_no_ambi', 'easy_yes_ambi', 'medium_no_ambi',
           'medium_yes_ambi', 'hard_no_ambi', 'hard_yes_ambi')

def count_records(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path}: expected nonempty list')
    seen = set()
    correct = 0
    for row in rows:
        if not isinstance(row, list) or len(row) != 4 or type(row[3]) is not int or row[3] not in (0, 1):
            raise ValueError(f'{path}: expected [scene_id, object_id, instruction, 0 or 1]')
        key = tuple(row[:3])
        if key in seen:
            raise ValueError(f'{path}: duplicate sample')
        seen.add(key)
        correct += row[3]
    return len(rows), correct

def summarize(counts):
    groups = {}
    for row in counts:
        key = (row['collection'], row['metric'], row['subset'])
        groups.setdefault(key, []).append(row)
    output = []
    for (collection, metric, subset), rows in sorted(groups.items()):
        splits = [str(r['split']) for r in rows]
        if len(splits) != len(set(splits)):
            raise ValueError(f'Duplicate split in {collection}/{subset}')
        vals = []
        for row in rows:
            n, k = int(row['total']), int(row['correct'])
            if n <= 0 or not 0 <= k <= n:
                raise ValueError('Invalid numerator/denominator')
            vals.append(k/n)
        complete = set(splits) == {'0', '1', '2'} and all(int(r['total']) == 50 for r in rows)
        output.append(dict(collection=collection, metric=metric, subset=subset,
                           n_splits=len(rows), complete_expected_3x50=complete,
                           total=sum(int(r['total']) for r in rows),
                           correct=sum(int(r['correct']) for r in rows),
                           mean=statistics.mean(vals) if complete else None,
                           population_sd=statistics.pstdev(vals) if complete else None,
                           sample_sd=statistics.stdev(vals) if complete else None,
                           observed_rates=vals,
                           note='All units are proportions. SD across instruction splits; not repeated seeds.'))
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--counts', type=Path, help='Exported per-split count CSV')
    source.add_argument('--raw-dir', type=Path, help='Directory with out_SUBSET_splitN.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.counts:
        with args.counts.open() as f:
            counts = list(csv.DictReader(f))
    else:
        counts = []
        for subset in SUBSETS:
            for split in range(3):
                p = args.raw_dir / f'out_{subset}_split{split}.json'
                if not p.exists():
                    continue
                n, k = count_records(p)
                counts.append(dict(collection='provided_raw', metric='recorded_segmentation_success',
                                   subset=subset, split=split, total=n, correct=k))
        if not counts:
            parser.error('No matching result records found')
    report = summarize(counts)
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
