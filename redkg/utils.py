import pickle
from typing import Any, Callable, Dict


class AttributeDict(dict):
    """Attributed dict class"""

    __getattr__: Callable = dict.__getitem__
    __setattr__: Callable = dict.__setitem__
    __delattr__: Callable = dict.__delitem__


def pickle_dump(file_path: str, file: Dict[Any, Any]) -> None:
    """Dump via pickle"""
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


def pickle_load(file_path: str) -> Any:
    """Load via pickle"""
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
    except EOFError:
        obj = None
    return obj
