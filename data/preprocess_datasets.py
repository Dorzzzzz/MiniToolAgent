import os
import json
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_from_disk

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[saved] {path}  ({len(records)} records)")


def process_aime25():
    ds = load_from_disk(str(RAW_DIR / "aime25"))
    split = ds["test"]
    records = [
        {
            "idx": int(row["id"]) + 1,
            "question": row["problem"],
            "answer": row["answer"],
            "gen_text_store": "",
        }
        for row in split
    ]
    write_jsonl(records, PROCESSED_DIR / "aime25.jsonl")


def process_hmmt_feb_2025():
    ds = load_from_disk(str(RAW_DIR / "hmmt_feb_2025"))
    split = ds["train"]
    records = [
        {
            "idx": row["problem_idx"],
            "question": row["problem"],
            "answer": row["answer"],
            "gen_text_store": "",
        }
        for row in split
    ]
    write_jsonl(records, PROCESSED_DIR / "hmmt_feb_2025.jsonl")


if __name__ == "__main__":
    process_aime25()
    process_hmmt_feb_2025()
    print("Preprocessing done.")
