from tqdm import tqdm
from typing import Iterable, Any


def progress_iter(i: Iterable[Any], show_progress: bool):
    return tqdm(i) if show_progress else i
