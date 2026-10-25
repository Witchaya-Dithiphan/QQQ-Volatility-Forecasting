import pytest

from src.ml.kernel_perceptron_test import main


def test_kernel_perceptron_test_requires_explicit_flag():
    with pytest.raises(PermissionError, match="--allow-test"):
        main([])
