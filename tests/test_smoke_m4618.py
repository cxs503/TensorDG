import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smoke_marker_m4618 import PIPELINE_SMOKE_M4618


def test_smoke_marker_value():
    assert PIPELINE_SMOKE_M4618 == "4618"


def test_basic_arithmetic():
    assert 1 + 1 == 2
