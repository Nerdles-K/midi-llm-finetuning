"""Create Pyramidk/midi-llm-merged repo on HF Hub and upload merged_model/."""
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "Pyramidk/midi-llm-merged"
LOCAL_DIR = Path(__file__).parent.parent / "merged_model"

api = HfApi()

print(f"Creating model repo {REPO_ID} (private=False, exist_ok=True)...")
create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
print("Repo ready.")

print(f"Uploading {LOCAL_DIR} to {REPO_ID} ...")
print("This will take 5-30 minutes depending on upload speed (~3.5 GB).")
api.upload_folder(
    folder_path=str(LOCAL_DIR),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload merged Llama-3.2-1B + LoRA (MIDI-LLM)",
)
print(f"Done. Model is at https://huggingface.co/{REPO_ID}")