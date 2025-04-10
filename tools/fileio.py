from pathlib import Path

def ensure_dir(path):
    """Create a dir recursively if it doesn't exist. 
    Returns True if the folder exists or was created."""
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path.is_dir()


def load_audio(path, ensure1d=False, **kwargs):
    """Load an audio file from the given path. 
    Returns a tuple (audio, sample_rate)"""
    import soundfile as sf
    kwargs.setdefault("dtype", "float32")
    kwargs.setdefault("always_2d", True)
    x, fs = sf.read(path, **kwargs)
    if ensure1d and x.ndim > 1:
        x = x[:, 0]
    return x.T, fs
