import os
import tempfile


def get_cache_filepath(name: str) -> str:
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, name)
    return file_path
