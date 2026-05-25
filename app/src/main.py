"""Process entry point that delegates startup to app.bootstrap.main."""
from __future__ import annotations
from app.bootstrap import main
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
