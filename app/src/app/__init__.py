"""Application package exports."""
from typing import Sequence

def main(argv: Sequence[str] | None=None) -> int:
    from .bootstrap import main as _main
    return _main(list(argv) if argv is not None else None)
__all__ = ['main']
