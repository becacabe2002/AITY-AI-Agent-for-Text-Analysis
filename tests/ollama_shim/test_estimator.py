"""Golden values from ADR-003 §12 (qwen3.5:4b, resident 4.2 GB, all 8 vCPUs busy)."""
import pytest

from conftest import SHIM_DIR
from estimator import allotted_vcpus, estimate, load_constants, read_cpu_usec

C = load_constants(SHIM_DIR / "sustainability_constants.toml")


def test_shipped_constants_v1():
    assert C["constants_version"] == 1
    assert (C["W_PER_VCPU"], C["W_PER_GB_RAM"], C["PUE"], C["GRID_KG_PER_KWH"]) == (
        3.125, 0.375, 1.08, 0.026)


@pytest.mark.parametrize("duration, cpu, kwh, kg", [
    (41, 328, 3.269e-4, 8.50e-6),        # short (~300 tokens)
    (456, 3648, 3.635e-3, 9.45e-5),      # long full paper
])
def test_golden_values(duration, cpu, kwh, kg):
    out = estimate(start_usec=0, end_usec=cpu * 1_000_000, duration_s=duration,
                   resident_gb=4.2, vcpus=8, constants=C)
    assert out["method"] == "cpu_time" and out["warnings"] == []
    assert out["energy_kwh"] == pytest.approx(kwh, rel=1e-3)
    assert out["co2e_kg"] == pytest.approx(kg, rel=1e-2)
    assert out["constants"]["MODEL_RESIDENT_GB"] == 4.2


def test_wall_clock_fallback_is_upper_bound():
    out = estimate(start_usec=None, end_usec=5, duration_s=456, resident_gb=4.2, vcpus=8,
                   constants=C, extra_warnings=["size from config"])
    assert out["method"] == "wall_clock" and out["cpu_seconds"] == 3648
    assert out["warnings"][0].startswith("cgroup cpu.stat unreadable")
    assert out["warnings"][1] == "size from config"


def test_counter_going_backwards_falls_back():
    out = estimate(start_usec=10, end_usec=5, duration_s=1, resident_gb=1, vcpus=2, constants=C)
    assert out["method"] == "wall_clock"


def test_read_cpu_usec(tmp_path):
    (tmp_path / "cpu.stat").write_text("usage_usec 123456\nuser_usec 100\n")
    assert read_cpu_usec(tmp_path) == 123456
    (tmp_path / "cpu.stat").write_text("user_usec 100\n")          # usage_usec line missing
    assert read_cpu_usec(tmp_path) is None
    assert read_cpu_usec(tmp_path / "missing") is None


def test_allotted_vcpus(tmp_path):
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    assert allotted_vcpus(tmp_path) == 4.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert allotted_vcpus(tmp_path) >= 1


def test_missing_constant_fails_at_boot(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("constants_version = 1\nPUE = 1.0\n")
    with pytest.raises(ValueError, match="missing constants"):
        load_constants(p)
