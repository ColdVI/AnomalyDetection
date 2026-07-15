"""Tests for Dashboard/codes/logging_setup.py.

ONEMLI: enable_file_logging() sys.stdout/sys.stderr'i GERCEKTEN degistirir --
her testten sonra orijinaline geri donduruluyor (fixture), aksi halde bu
dosyadaki bir test sonraki testlerin ciktisini sessizce yutabilir."""

from __future__ import annotations

import sys

import pytest

import Dashboard.codes.logging_setup as logging_setup
from Dashboard.codes.logging_setup import enable_file_logging


@pytest.fixture(autouse=True)
def _restore_stdio():
    orig_out, orig_err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = orig_out, orig_err


def test_enable_file_logging_creates_logs_dir_and_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enable_file_logging("svc", logs_dir="logs")

    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "logs" / "svc.log").exists()


def test_enable_file_logging_default_ignores_cwd_uses_dashboard_logs(tmp_path, monkeypatch):
    """Regresyon: logs_dir varsayilani onceden CALISMA DIZININE gore bagliydi
    (relative "logs") -- script nereden calistirilirsa calistirilsin farkli/
    beklenmeyen bir yerde logs/ olusuyordu. Varsayilan artik cwd'den
    BAGIMSIZ, Dashboard/logs/'e sabit -- burada gercek Dashboard/logs/'e
    yazmamak icin _DEFAULT_LOGS_DIR bir tmp_path'e monkeypatch'leniyor."""
    fake_default = tmp_path / "Dashboard" / "logs"
    monkeypatch.setattr(logging_setup, "_DEFAULT_LOGS_DIR", fake_default)
    (tmp_path / "somewhere" / "else").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "somewhere" / "else")

    enable_file_logging("svc")  # logs_dir verilmiyor -- varsayilan kullanilmali

    assert fake_default.is_dir()
    assert (fake_default / "svc.log").exists()
    assert not (tmp_path / "somewhere" / "else" / "logs").exists()


def test_enable_file_logging_writes_print_output_to_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enable_file_logging("svc", logs_dir="logs")

    print("merhaba")
    sys.stdout.flush()

    assert (tmp_path / "logs" / "svc.log").read_text(encoding="utf-8") == "merhaba\n"


def test_enable_file_logging_still_writes_to_real_stdout(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    enable_file_logging("svc", logs_dir="logs")

    print("hem terminalde hem dosyada gorunmeli")
    sys.stdout.flush()

    captured = capfd.readouterr()
    assert "hem terminalde hem dosyada gorunmeli" in captured.out


def test_enable_file_logging_appends_across_multiple_calls(tmp_path, monkeypatch):
    """Servis yeniden baslarsa (restart: unless-stopped) eski log SILINMEMELI."""
    monkeypatch.chdir(tmp_path)

    enable_file_logging("svc", logs_dir="logs")
    print("birinci calisma")
    sys.stdout.flush()

    enable_file_logging("svc", logs_dir="logs")
    print("ikinci calisma")
    sys.stdout.flush()

    content = (tmp_path / "logs" / "svc.log").read_text(encoding="utf-8")
    assert "birinci calisma" in content
    assert "ikinci calisma" in content


def test_enable_file_logging_rotates_oversized_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    big_file = tmp_path / "logs" / "svc.log"
    big_file.write_bytes(b"x" * (21 * 1024 * 1024))  # 21MB > _MAX_BYTES esigi

    enable_file_logging("svc", logs_dir="logs")

    assert (tmp_path / "logs" / "svc.log.1").stat().st_size > 20 * 1024 * 1024
    assert (tmp_path / "logs" / "svc.log").stat().st_size == 0


def test_enable_file_logging_does_not_rotate_small_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    small_file = tmp_path / "logs" / "svc.log"
    small_file.write_text("kucuk dosya")

    enable_file_logging("svc", logs_dir="logs")

    assert not (tmp_path / "logs" / "svc.log.1").exists()
    assert "kucuk dosya" in (tmp_path / "logs" / "svc.log").read_text(encoding="utf-8")
