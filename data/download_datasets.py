import os                                                                                                   
from pathlib import Path
                                                                                                              
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"       

from datasets import load_dataset
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("MathArena/hmmt_feb_2025", "hmmt_feb_2025"),
    ("math-ai/aime25", "aime25"),
]

for repo_id, name in DATASETS:
    save_path = RAW_DIR / name
    if save_path.exists():
        print(f"[skip] {name} already exists at {save_path}")
        continue
    print(f"[download] {repo_id} -> {save_path}")
    ds = load_dataset(repo_id)
    ds.save_to_disk(str(save_path))
    print(f"[done] {name}")

print("All datasets downloaded.")