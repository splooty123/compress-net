from PIL import Image
from pathlib import Path

folder = Path("images")

for p in folder.iterdir():
    if p.is_file() and p.suffix.lower() != ".bmp":
        try:
            out = p.with_suffix(".bmp")
            Image.open(p).convert("RGB").save(out)
            print(f"{p.name} -> {out.name}")
            p.unlink(missing_ok=True)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
