from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_dir(model_path_or_id: str) -> Path:
    """Resolves `model_path_or_id` and downloads the model from Hugging
    Face if not found locally.

    Parameters
    ----------
    model_path_or_id : str
        Path to local model checkpoint directory or Hugging Face repo ID

    Returns
    -------
    Path
        The resolved absolute path to the local model checkpoint directory
    """
    path = Path(model_path_or_id)
    return path if path.exists() else Path(snapshot_download(model_path_or_id))
