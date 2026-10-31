import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

METHODS = ("chen", "synthid_seq", "synthid_struct", "foldmark")
OPS = ("blosum62", "proteinmpnn", "esm2")
IDENTITIES = (100, 95, 90, 80)


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def output_file(path):
    """Publish a file only after its entire stage succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, newline="") as handle:
        tmp = Path(handle.name)
        try:
            yield handle
            handle.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
