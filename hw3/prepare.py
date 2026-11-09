import csv

from pathlib import Path

CAPTIONS_PATH = Path("test.csv")
OUTPUT_PATH = Path("outputs/av.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with CAPTIONS_PATH.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with OUTPUT_PATH.open("x", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "caption"])
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "name": row["id"],
            "caption": row["caption"],
        })

print(f"Created {OUTPUT_PATH}")
