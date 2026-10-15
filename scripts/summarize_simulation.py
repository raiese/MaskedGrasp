"""Summarize historical case files; do not assign methods from numeric similarity."""
import argparse
import json
import statistics
from pathlib import Path

def read_case(path):
    tokens = path.read_text().split()
    if len(tokens) < 5:
        raise ValueError(f'Malformed result: {path.name}')
    success, steps, successful_steps, reward = map(float, tokens[-4:])
    if not 0 <= success <= 1 or steps < 0:
        raise ValueError(f'Invalid result: {path.name}')
    return dict(case=path.stem, instruction=' '.join(tokens[:-4]), success_rate=success,
                mean_steps=steps, mean_successful_steps=successful_steps if success > 0 else None,
                stored_successful_steps=successful_steps, mean_reward=reward)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True,
                        help='Parent of RUN/results/caseN.txt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = []
    for root in sorted(args.results_root.glob('*/results')):
        rows = [read_case(p) for p in sorted(root.glob('case*.txt'))]
        if not rows:
            continue
        valid = [r['mean_successful_steps'] for r in rows if r['mean_successful_steps'] is not None]
        report.append(dict(run=root.parent.name, method='unassigned', cases=len(rows),
                           macro_success_rate=statistics.mean(r['success_rate'] for r in rows),
                           macro_steps=statistics.mean(r['mean_steps'] for r in rows),
                           macro_successful_steps_nonempty_cases=statistics.mean(valid) if valid else None,
                           pooled_successful_steps=None,
                           note='Pooled success-conditioned mean requires per-episode counts. Zero-success sentinel 1000 excluded.',
                           per_case=rows))
    if not report:
        parser.error('No case result files found')
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
