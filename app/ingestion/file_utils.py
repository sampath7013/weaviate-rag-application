import hashlib
from pathlib import Path


def calculate_file_hash(
    file_path: str,
    chunk_size: int = 8192,
) -> str:

    sha256 = hashlib.sha256()

    path = Path(file_path)

    with path.open("rb") as file:

        while True:

            data = file.read(chunk_size)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()