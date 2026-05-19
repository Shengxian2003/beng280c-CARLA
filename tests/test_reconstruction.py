"""Unit tests for the reconstruction skill (Stage 2a).

These tests cover the Python-side glue: path translation, config dataclass,
MATLAB command construction, and validation. They do NOT invoke MATLAB —
the live MATLAB integration is exercised by the smoke test in
notebooks/test_reconstruction_skill.py.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from skills.reconstruction import (
    reconstruct,
    save_preview,
    ReconConfig,
    to_windows_for_matlab,
    to_wsl,
    wsl_to_unc,
    mnt_to_windows,
)
from skills.reconstruction._paths import is_wsl_path, is_mnt_path, windows_to_mnt
from skills.reconstruction._runner import _build_matlab_cmd


# ---------- Path translation -------------------------------------------------

class TestPathDetection:
    def test_wsl_path_detected(self):
        assert is_wsl_path("/home/nick_17/projects/medict/data.mat")
        assert is_wsl_path("/etc/passwd")

    def test_mnt_path_not_wsl(self):
        assert not is_wsl_path("/mnt/g/medict_tmp/data.mat")

    def test_windows_path_not_wsl(self):
        assert not is_wsl_path("G:\\medict_tmp\\data.mat")

    def test_mnt_path_detected(self):
        assert is_mnt_path("/mnt/g/data.mat")
        assert is_mnt_path("/mnt/c/Users/x/data.mat")

    def test_non_mnt_paths_rejected(self):
        assert not is_mnt_path("/home/nick/data.mat")
        assert not is_mnt_path("/mnt/foo/bar")  # /mnt/foo is too short for a drive letter
        assert not is_mnt_path("G:\\data.mat")


class TestWslToUnc:
    def test_basic_home_path(self):
        result = wsl_to_unc("/home/nick_17/data.mat")
        assert result == "\\\\wsl.localhost\\Ubuntu\\home\\nick_17\\data.mat"

    def test_custom_distro(self):
        result = wsl_to_unc("/home/x/y", distro="Debian")
        assert result == "\\\\wsl.localhost\\Debian\\home\\x\\y"

    def test_root_only(self):
        result = wsl_to_unc("/foo")
        assert result == "\\\\wsl.localhost\\Ubuntu\\foo"

    def test_relative_path_raises(self):
        with pytest.raises(ValueError):
            wsl_to_unc("home/nick_17/data.mat")


class TestMntToWindows:
    def test_basic_g_drive(self):
        assert mnt_to_windows("/mnt/g/medict_tmp/data.mat") == "G:\\medict_tmp\\data.mat"

    def test_c_drive_uppercased(self):
        assert mnt_to_windows("/mnt/c/Users/x") == "C:\\Users\\x"

    def test_drive_root(self):
        assert mnt_to_windows("/mnt/g/") == "G:\\"

    def test_non_mnt_raises(self):
        with pytest.raises(ValueError):
            mnt_to_windows("/home/x/y")


class TestWindowsToMnt:
    def test_basic(self):
        assert windows_to_mnt("G:\\medict_tmp\\data.mat") == "/mnt/g/medict_tmp/data.mat"

    def test_no_drive_raises(self):
        with pytest.raises(ValueError):
            windows_to_mnt("medict_tmp\\data.mat")


class TestToWindowsForMatlab:
    def test_mnt_becomes_native(self):
        # /mnt/* should become X:\ — preferred for data files
        assert to_windows_for_matlab("/mnt/g/x.mat") == "G:\\x.mat"

    def test_wsl_becomes_unc(self):
        # /home/* should become UNC — fine for .m source
        assert to_windows_for_matlab("/home/nick/x.m").startswith("\\\\wsl.localhost\\")

    def test_windows_path_unchanged(self):
        assert to_windows_for_matlab("G:\\x.mat") == "G:\\x.mat"


class TestToWsl:
    def test_windows_drive_to_mnt(self):
        assert to_wsl("G:\\x.mat") == "/mnt/g/x.mat"

    def test_unc_to_wsl_path(self):
        assert to_wsl("\\\\wsl.localhost\\Ubuntu\\home\\x") == "/home/x"

    def test_already_wsl_unchanged(self):
        assert to_wsl("/home/x.mat") == "/home/x.mat"


# ---------- Config dataclass -------------------------------------------------

class TestReconConfig:
    def test_serializable(self):
        cfg = ReconConfig(
            input_mat="G:\\in.mat", output_mat="G:\\out.mat",
            recon_dir="\\\\wsl.localhost\\Ubuntu\\foo",
            method="cs", is_flow=1, is_rest=1,
            n_iterations=5, n_coils=12, use_gpu=1,
            data_field="D", venc_m_per_s=1.5,
        )
        # Must round-trip through JSON cleanly for the MATLAB driver
        json_str = json.dumps(asdict(cfg))
        parsed = json.loads(json_str)
        assert parsed["method"] == "cs"
        assert parsed["n_iterations"] == 5
        assert parsed["venc_m_per_s"] == 1.5


# ---------- MATLAB command builder ------------------------------------------

class TestStreamSubprocess:
    """The on_line callback fires for every subprocess line — used by the agent
    tool wrapper to show a live elapsed-time counter during long MATLAB runs."""

    def test_on_line_called_per_line(self):
        from skills.reconstruction._runner import _stream_subprocess
        captured = []
        # Cross-platform: use python -c with three prints
        rc = _stream_subprocess(
            ["python", "-c", "print('one'); print('two'); print('three')"],
            verbose=False,
            on_line=captured.append,
        )
        assert rc == 0
        assert captured == ["one", "two", "three"]

    def test_on_line_optional(self):
        from skills.reconstruction._runner import _stream_subprocess
        rc = _stream_subprocess(
            ["python", "-c", "print('x')"], verbose=False,
        )
        assert rc == 0


class TestBuildMatlabCmd:
    def test_command_shape(self):
        cmd = _build_matlab_cmd(
            matlab_exe="/mnt/g/Application_Industry/Matlab/bin/matlab.exe",
            driver_dir_win="G:\\medict_tmp",
            config_path_win="G:\\medict_tmp\\config.json",
        )
        assert len(cmd) == 3
        assert cmd[0].endswith("matlab.exe")
        assert cmd[1] == "-batch"

    def test_batch_body_calls_driver(self):
        cmd = _build_matlab_cmd(
            matlab_exe="matlab.exe",
            driver_dir_win="G:\\skills\\reconstruction",
            config_path_win="G:\\medict_tmp\\config.json",
        )
        body = cmd[2]
        assert "addpath('G:\\skills\\reconstruction')" in body
        assert "medict_recon_driver('G:\\medict_tmp\\config.json')" in body

    def test_single_quote_escaped(self):
        # If a path contains a single quote, it must be escaped as '' in MATLAB
        cmd = _build_matlab_cmd(
            matlab_exe="matlab.exe",
            driver_dir_win="G:\\dir'with'quote",
            config_path_win="G:\\cfg.json",
        )
        # The escaped form appears as '' between letters
        assert "dir''with''quote" in cmd[2]


# ---------- Validation in reconstruct() --------------------------------------

class TestReconstructValidation:
    def test_unknown_method_rejected(self, tmp_path):
        fake = tmp_path / "ks.mat"
        fake.touch()
        with pytest.raises(ValueError, match="method must be"):
            reconstruct(fake, method="bogus")

    def test_zero_iterations_rejected(self, tmp_path):
        fake = tmp_path / "ks.mat"
        fake.touch()
        with pytest.raises(ValueError, match="n_iterations"):
            reconstruct(fake, n_iterations=0)

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            reconstruct(tmp_path / "does_not_exist.mat")


# ---------- Driver script presence ------------------------------------------

class TestDriverArtifacts:
    def test_driver_m_exists(self):
        from skills.reconstruction._runner import _DRIVER_M, _RECON_DIR
        assert _DRIVER_M.exists(), \
            f"medict_recon_driver.m missing at {_DRIVER_M}"
        # The motion-robust-CMR code directory should be present
        assert _RECON_DIR.exists(), \
            f"motion-robust-CMR recon dir missing at {_RECON_DIR}"

    def test_driver_function_name_matches(self):
        from skills.reconstruction._runner import _DRIVER_M
        first_line = _DRIVER_M.read_text().splitlines()[0]
        # MATLAB requires function name to match file name
        assert "function medict_recon_driver" in first_line


# ---------- Preview generation (synthetic data, no MATLAB) ------------------

def _synthetic_result(Z=8, Y=12, X=10, T=4):
    rng = np.random.default_rng(0)
    xHat = (rng.normal(size=(Z, Y, X, T)) + 1j * rng.normal(size=(Z, Y, X, T))).astype(np.complex128)
    tx = rng.uniform(-np.pi, np.pi, size=(Z, Y, X, T)).astype(np.float64)
    ty = rng.uniform(-np.pi, np.pi, size=(Z, Y, X, T)).astype(np.float64)
    tz = rng.uniform(-np.pi, np.pi, size=(Z, Y, X, T)).astype(np.float64)
    return {"xHat": xHat, "thetaX": tx, "thetaY": ty, "thetaZ": tz,
            "meta": {"venc_m_per_s": 1.5}}


class TestSavePreview:
    def test_writes_four_pngs(self, tmp_path):
        paths = save_preview(_synthetic_result(), tmp_path)
        assert len(paths) == 4
        for p in paths:
            assert p.exists() and p.stat().st_size > 0
        names = {p.name for p in paths}
        assert names == {
            "1_anatomy.png", "2_speed.png",
            "3_velocity_components.png", "4_max_speed_projection.png",
        }

    def test_creates_missing_dir(self, tmp_path):
        sub = tmp_path / "nested" / "preview"
        save_preview(_synthetic_result(), sub)
        assert sub.is_dir()

    def test_venc_override(self, tmp_path):
        # passing venc explicitly must not fall back to meta
        r = _synthetic_result()
        r["meta"]["venc_m_per_s"] = 99.0  # bogus — should be ignored
        paths = save_preview(r, tmp_path, venc_m_per_s=1.5)
        assert all(p.exists() for p in paths)

    def test_accepts_complex_xhat(self, tmp_path):
        # xHat from real recons comes back complex; preview must abs() it
        r = _synthetic_result()
        assert np.iscomplexobj(r["xHat"])
        save_preview(r, tmp_path)  # must not raise
