"""Trusted suite-wide audit boundary, active before test collection."""
import pytest
from src.modeling.test_access import install_test_access_guard, access_counts

install_test_access_guard()

def pytest_sessionfinish(session, exitstatus):
    counts = access_counts()
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    if reporter:
        reporter.write_sep('=', f'Accepted Test audit: {counts}')
    if counts['blocked'] or counts['accepted']:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
