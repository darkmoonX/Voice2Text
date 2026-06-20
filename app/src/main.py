"""Process entry point that delegates startup to voice2text.bootstrap.main."""
from __future__ import annotations
from voice2text.bootstrap import main
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
