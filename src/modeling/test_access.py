"""Process audit boundary for Train/Validation work; never grants Test authorization.

Install before importing training/data code. Denied attempts are counted before I/O.
Synthetic fixtures outside repository data roots remain available to tests.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from config import PROJECT_ROOT

_installed = False
_blocked = 0
_ROOT = os.path.normcase(os.path.realpath(PROJECT_ROOT))

def is_accepted_test_path(value) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    path = os.path.normcase(os.path.realpath(os.fsdecode(value)))
    try:
        relative = Path(path).relative_to(_ROOT)
    except ValueError:
        return False
    parts = tuple(part.lower() for part in relative.parts)
    protected_root = bool(parts) and (parts[0] == 'data' or 'accepted-reproduction' in parts or parts[:2] == ('outputs', 'experiments'))
    return protected_root and ((relative.suffix.lower() == '.csv' and 'test' in relative.name.lower()) or any(part in ('test', 'test_split') for part in parts))

def _audit(event, args):
    global _blocked
    if event in ('open', 'os.listdir', 'os.scandir'):
        paths = args[:1]
    elif event in ('shutil.copyfile', 'shutil.copytree'):
        paths = args[:2]
    else:
        return
    if any(is_accepted_test_path(path) for path in paths):
        _blocked += 1
        raise PermissionError('Accepted Test access is forbidden during Train/Validation verification')

def install_test_access_guard() -> None:
    global _installed
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True

def access_counts() -> dict[str, int]:
    return {'blocked': _blocked, 'accepted': 0}
