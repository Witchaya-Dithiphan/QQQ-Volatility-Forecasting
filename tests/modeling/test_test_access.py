"""Guard contract: tests use isolated subprocesses for deliberate denials."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from config import PROJECT_ROOT

@pytest.mark.parametrize('operation', ['open', 'copy', 'hash'])
def test_guard_blocks_and_counts_before_io(tmp_path, operation):
    code = '''
import json, shutil, hashlib
from pathlib import Path
from src.modeling.test_access import install_test_access_guard, access_counts
from config import PROJECT_ROOT
install_test_access_guard()
p = PROJECT_ROOT / 'data/processed/test_labeled.csv'
try:
    if OP == 'open': p.open('rb')
    elif OP == 'copy': shutil.copyfile(p, DEST)
    else: hashlib.sha256(p.read_bytes())
except PermissionError:
    print(json.dumps(access_counts()))
else:
    raise AssertionError('guard allowed accepted Test access')
'''.replace('OP', repr(operation)).replace('DEST', repr(str(tmp_path / 'copy.csv')))
    result = subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'blocked': 1, 'accepted': 0}
    assert not (tmp_path / 'copy.csv').exists()

def test_guard_allows_train_validation_and_synthetic_test(tmp_path):
    from src.modeling.test_access import install_test_access_guard, access_counts
    install_test_access_guard()
    before = access_counts()
    for name in ('train.csv', 'validation.csv', 'test.csv'):
        path = tmp_path / name
        path.write_bytes(b'synthetic')
        assert path.read_bytes() == b'synthetic'
    assert access_counts() == before

@pytest.mark.parametrize('relative', ['data/processed/test.csv', 'data/processed/test_labeled.csv', 'data/reproduction/accepted-reproduction/data/processed/test_labeled.csv', 'data/experiments/diagnostics/test_flagged.csv'])
def test_guard_classifies_accepted_paths_without_opening(relative):
    from src.modeling.test_access import is_accepted_test_path
    assert is_accepted_test_path(PROJECT_ROOT / relative)
    assert not is_accepted_test_path(PROJECT_ROOT / 'data/processed/train_labeled.csv')

def test_normal_cli_installs_guard_before_training_imports():
    import ast
    tree = ast.parse((PROJECT_ROOT / 'src/modeling/runner.py').read_text())
    calls = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'install_test_access_guard']
    imports = [i for i, node in enumerate(tree.body) if isinstance(node, ast.ImportFrom) and node.module == 'datasets']
    assert len(calls) == 1
    assert calls[0] < min(imports)
