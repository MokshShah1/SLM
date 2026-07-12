# One-shot: publish dataset + model card to the Hub. Run in Colab or anywhere you're logged in.
#   pip install huggingface_hub datasets
#   huggingface-cli login    (paste your token)
#   python _upload.py
from huggingface_hub import HfApi, upload_file
api = HfApi()
USER = "mokshpshah"

# 1) dataset repo
ds_repo = f"{USER}/socratic-tutor-dataset"
api.create_repo(ds_repo, repo_type="dataset", exist_ok=True)
for f in ["data/train.jsonl", "data/val.jsonl", "eval/tutor_eval.jsonl", "SAMPLE_DATASET.md"]:
    upload_file(path_or_fileobj=f, path_in_repo=f.split("/")[-1],
                repo_id=ds_repo, repo_type="dataset")
print("dataset ->", f"https://huggingface.co/datasets/{ds_repo}")

# 2) model card onto the existing model repo
upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md",
            repo_id=f"{USER}/qwen3-1.7b-socratic-tutor", repo_type="model")
print("model card -> https://huggingface.co/" + USER + "/qwen3-1.7b-socratic-tutor")
