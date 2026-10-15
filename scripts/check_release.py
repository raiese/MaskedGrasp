"""Check the local code candidate for accidental large/private artifacts (no network)."""
import argparse
import ast
import json
import re
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    findings=[]
    code_count=0
    patterns={
        'credential-like token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})'),
        'private home path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
        'private network address': re.compile(r'\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b'),
        'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    }
    banned={'.npz','.npy','.parquet','.safetensors','.pth','.pt','.tar','.zip','.pdf','.so','.pyc'}
    files=[f for f in args.root.rglob('*') if f.is_file() and '.git' not in f.parts]
    for f in files:
        rel=str(f.relative_to(args.root))
        if f.is_symlink(): findings.append(dict(file=rel,reason='symlink'))
        if f.suffix in banned or f.stat().st_size>5_000_000:
            findings.append(dict(file=rel,reason='binary/large artifact'))
            continue
        try: text=f.read_text()
        except UnicodeDecodeError:
            findings.append(dict(file=rel,reason='non-text artifact'));continue
        for reason,pattern in patterns.items():
            if pattern.search(text): findings.append(dict(file=rel,reason=reason))
        if f.suffix=='.py':
            code_count+=1
            try: ast.parse(text,filename=rel)
            except SyntaxError as exc: findings.append(dict(file=rel,reason=f'syntax error line {exc.lineno}'))
    print(json.dumps(dict(files=len(files),python_files=code_count,
                          bytes=sum(f.stat().st_size for f in files),findings=findings),indent=2))
    raise SystemExit(bool(findings))

if __name__=='__main__': main()
