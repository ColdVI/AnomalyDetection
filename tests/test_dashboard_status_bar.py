"""Dashboard/app.py -- durum cubugu (üst orta overlay) metin formati ve
yerlesim testleri.

ONEMLI: test_status_div_children_order_regression, kullanicinin acikca
istedigi "gösterilen'le aktif uçuş yan yana olsun, alarmı en sağa al"
duzenini dogrudan layout agacindan (regex/string DEGIL, GERCEK Dash
bilesen nesnesi) dogrulayan bir regresyon testidir."""

from __future__ import annotations

from Dashboard import app as dashapp
from dashboard_fakes import find_by_id


# ------------------------------------------------------ status_bar_* format --

def test_status_bar_main_format_tr():
    result = dashapp.TEXTS["tr"]["status_bar_main"].format(ts="10:00:00", n=42)
    assert result == "10:00:00 | 42 aktif uçuş"


def test_status_bar_alarm_format_tr():
    assert dashapp.TEXTS["tr"]["status_bar_alarm"].format(a=3) == " | 3 alarm"


def test_status_bar_alarm_zero_still_shown():
    """0 alarm bile gosterilmeli (sessizce gizlenmemeli) -- kullanici her
    zaman "0 alarm" ile aktif izleniyor oldugunu teyit edebilmeli."""
    assert "0" in dashapp.TEXTS["tr"]["status_bar_alarm"].format(a=0)


# ------------------------------------------------------- layout yerlesim sirasi --

def test_status_div_children_order_regression():
    """Kullanici istegi: 'gösterilenle aktif uçuş yan yana olsun, alarmı en
    sağa al' -- DOM sirasi = gorsel sira, bu yuzden dogru sira TAM OLARAK
    [status-main, status-shown, status-alarm] olmali."""
    status_div = find_by_id(dashapp.app_dash.layout, "status")
    assert status_div is not None
    child_ids = [getattr(c, "id", None) for c in status_div.children]
    assert child_ids == ["status-main", "status-shown", "status-alarm"]


def test_status_shown_span_is_visually_distinct_color():
    """'gösteriliyor' sayisi (firma filtresi dahil nihai sayi) diger
    metinden renkle ayirt edilebilir olmali (bkz. proje sohbet gecmisi)."""
    status_div = find_by_id(dashapp.app_dash.layout, "status")
    shown_span = next(c for c in status_div.children if c.id == "status-shown")
    shown_color = shown_span.style.get("color")
    default_text_color = status_div.style.get("color")
    assert shown_color is not None
    assert shown_color != default_text_color
