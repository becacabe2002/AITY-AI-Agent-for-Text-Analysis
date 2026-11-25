"""Raw cgroup readings of the shim (the formula is tested in tests/aity_web/test_aity_cpu_model.py)."""
from readings import allotted_vcpus, cpu_seconds_between, read_cpu_usec


def test_read_cpu_usec(tmp_path):
    (tmp_path / "cpu.stat").write_text("usage_usec 123456\nuser_usec 100\n")
    assert read_cpu_usec(tmp_path) == 123456
    (tmp_path / "cpu.stat").write_text("user_usec 100\n")          # usage_usec line missing
    assert read_cpu_usec(tmp_path) is None
    assert read_cpu_usec(tmp_path / "missing") is None


def test_cpu_seconds_between():
    assert cpu_seconds_between(1_000_000, 3_500_000) == 2.5
    assert cpu_seconds_between(None, 5) is None
    assert cpu_seconds_between(5, None) is None
    assert cpu_seconds_between(10, 5) is None                       # counter went backwards


def test_allotted_vcpus(tmp_path):
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    assert allotted_vcpus(tmp_path) == 4.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert allotted_vcpus(tmp_path) >= 1
