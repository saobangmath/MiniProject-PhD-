"""Add project root to sys.path so notebooks can `import libs`."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "libs" / "__init__.py").is_file():
            root = candidate.resolve()
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return root
    raise RuntimeError("Could not locate project root (libs/__init__.py)")


add_project_root()
