"""Filesystem code identity coverage, independent of Git and ignored status."""
import hashlib
from pathlib import Path
import pytest
from src.modeling.artifacts import code_snapshot_hash

@pytest.fixture
def source_root(tmp_path):
    for name in ('config.py', 'configs/modeling.json', 'src/__init__.py', 'src/modeling/a.py', 'src/models/b.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return tmp_path

def test_tracked_source_mutation(source_root):
    before = code_snapshot_hash(source_root)
    (source_root / 'src/models/b.py').write_text('changed')
    assert code_snapshot_hash(source_root) != before

@pytest.mark.parametrize('name', ['src/modeling/untracked.py', 'src/models/ignored.py', 'src/modeling/test_access.py', 'src/__init__.py'])
def test_runtime_source_addition_and_mutation(source_root, name):
    before = code_snapshot_hash(source_root)
    path = source_root / name
    path.write_text('new source')
    added = code_snapshot_hash(source_root)
    assert added != before
    path.write_text('changed source')
    assert code_snapshot_hash(source_root) != added

@pytest.mark.parametrize('name', ['outputs/generated.py', '.venv/source.py', 'tmp/sitecustomize.py', '.pytest_cache/cache.py', 'src/modeling/__pycache__/cached.pyc'])
def test_irrelevant_files_do_not_count(source_root, name):
    before = code_snapshot_hash(source_root)
    path = source_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('irrelevant')
    assert code_snapshot_hash(source_root) == before

def test_deterministic_posix_path_order(source_root, monkeypatch):
    paths = [source_root / 'config.py', source_root / 'configs/modeling.json', source_root / 'src/__init__.py']
    paths += list((source_root / 'src/modeling').rglob('*.py')) + list((source_root / 'src/models').rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(source_root).as_posix()):
        data = path.read_bytes()
        digest.update(path.relative_to(source_root).as_posix().encode() + b'\0' + len(data).to_bytes(8, 'big') + data)
    expected = digest.hexdigest()
    original = Path.rglob
    monkeypatch.setattr(Path, 'rglob', lambda self, pattern: iter(reversed(list(original(self, pattern)))))
    assert code_snapshot_hash(source_root) == expected
    assert code_snapshot_hash(source_root / '.') == expected
    assert code_snapshot_hash(source_root) == code_snapshot_hash(source_root)
