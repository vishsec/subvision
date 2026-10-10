from pathlib import Path
root = Path("data")
for sub in sorted([d for d in root.rglob("*") if d.is_dir()]):
    n = len(list(sub.glob("*.png"))) + len(list(sub.glob("*.jpg"))) + len(list(sub.glob("*.jpeg")))
    if n: print(sub.relative_to(root), n)