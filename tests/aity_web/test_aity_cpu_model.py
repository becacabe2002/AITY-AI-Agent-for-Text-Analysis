"""Backend CPU-time energy estimate (ADR-003 §4). Golden values from ADR-003 §12
(qwen3.5:4b, resident 4.2 GB, all 8 vCPUs busy)."""
import pytest

from aity_web.services import aity_cpu_model as m

C = m.load_constants()


def _readings(**over):
    return {"cpu_seconds": 328.0, "duration_s": 41.0, "vcpus": 8.0,
            "resident_bytes": 4_200_000_000, "model": "qwen3.5:4b"} | over


def test_shipped_constants_v1():
    assert C["constants_version"] == 1
    assert (C["W_PER_VCPU"], C["W_PER_GB_RAM"], C["PUE"], C["GRID_KG_PER_KWH"]) == (
        3.125, 0.375, 1.08, 0.026)
    assert C["model_resident_gb"]["qwen3.5:4b"] == 4.2


@pytest.mark.parametrize("duration, cpu, kwh, kg", [
    (41, 328, 3.269e-4, 8.50e-6),        # short (~300 tokens)
    (456, 3648, 3.635e-3, 9.45e-5),      # long full paper
])
def test_golden_values(duration, cpu, kwh, kg):
    est = m.estimate(_readings(cpu_seconds=cpu, duration_s=duration), C)
    assert est.method == "cpu_time" and est.warnings == ()
    assert est.energy_kwh == pytest.approx(kwh, rel=1e-3)
    assert est.co2e_kg == pytest.approx(kg, rel=1e-2)
    assert est.resident_gb == 4.2


def test_inputs_entry_records_everything():
    est = m.estimate(_readings(cpu_seconds=3648.0, duration_s=456.0), C)
    assert est.inputs_entry() == (
        "aity_cpu_model v1: method=cpu_time, cpu_seconds=3648.0, duration_s=456.0, "
        "W_PER_VCPU=3.125, W_PER_GB_RAM=0.375, PUE=1.08, GRID_KG_PER_KWH=0.026, "
        "MODEL_RESIDENT_GB=4.2"
    )


def test_wall_clock_fallback_is_upper_bound():
    est = m.estimate(_readings(cpu_seconds=None, duration_s=456.0), C)
    assert est.method == "wall_clock" and est.cpu_seconds == 3648.0
    assert est.warnings[0].startswith("cgroup cpu.stat unreadable")


@pytest.mark.parametrize("model, gb", [("qwen3.5:4b", 4.2), ("gemma3:4b", 4.2), (None, 4.2)])
def test_resident_size_falls_back_to_config(model, gb):
    est = m.estimate(_readings(resident_bytes=None, model=model), C)
    assert est.resident_gb == gb
    assert est.warnings == (f"resident model size not available from /api/ps; used config value {gb} GB",)


def test_per_model_config_size_is_used(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text((m.DEFAULT_CONSTANTS_FILE.read_text(encoding="utf-8"))
                 + '"gemma3:4b" = 3.3\n')
    est = m.estimate(_readings(resident_bytes=None, model="gemma3:4b"), m.load_constants(p))
    assert est.resident_gb == 3.3


@pytest.mark.parametrize("bad", [
    {"duration_s": None}, {"duration_s": -1}, {"duration_s": "41"}, {"duration_s": True},
    {"cpu_seconds": -1.0}, {"cpu_seconds": "328"},
    {"cpu_seconds": None, "vcpus": 0}, {"cpu_seconds": None, "vcpus": None},
    {"resident_bytes": -5},
])
def test_malformed_readings_raise(bad):
    with pytest.raises(m.ReadingsError):
        m.estimate(_readings(**bad), C)


def test_non_mapping_raises():
    with pytest.raises(m.ReadingsError):
        m.estimate(["not", "a", "dict"], C)


def test_missing_constant_fails(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("constants_version = 1\nPUE = 1.0\n")
    with pytest.raises(m.ConstantsError, match="missing constants"):
        m.load_constants(p)


@pytest.mark.parametrize("text, match", [
    ('constants_version = "1"\nW_PER_VCPU = 1\nW_PER_GB_RAM = 1\nPUE = 1\nGRID_KG_PER_KWH = 1\n'
     'DEFAULT_MODEL_RESIDENT_GB = 1\n', "constants_version"),
    ('constants_version = 1\nW_PER_VCPU = -1\nW_PER_GB_RAM = 1\nPUE = 1\nGRID_KG_PER_KWH = 1\n'
     'DEFAULT_MODEL_RESIDENT_GB = 1\n', "W_PER_VCPU"),
    ("not toml [", "cannot read"),
])
def test_bad_constants_files(tmp_path, text, match):
    p = tmp_path / "c.toml"
    p.write_text(text)
    with pytest.raises(m.ConstantsError, match=match):
        m.load_constants(p)


def test_env_var_overrides_location(tmp_path, monkeypatch):
    p = tmp_path / "c.toml"
    p.write_text(m.DEFAULT_CONSTANTS_FILE.read_text(encoding="utf-8").replace(
        "constants_version = 1", "constants_version = 2"))
    monkeypatch.setenv(m.CONSTANTS_ENV, str(p))
    assert m.load_constants()["constants_version"] == 2


def test_default_constants_cached():
    m.default_constants.cache_clear()
    assert m.default_constants() is m.default_constants()
