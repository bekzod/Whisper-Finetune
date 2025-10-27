from huggingface_hub import snapshot_download

repo_id = "bekzod123/whisper-v3-large-best"
revision = "main"  # or whichever branch/commit you want
local_dir = "./checkpoint-final"  # where you want to save

# Use allow_patterns to only download the folder “checkpoint-final”
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=["checkpoint-final/*"],
)
