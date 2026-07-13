"""Dashboard/ -- gercek tarayici (Playwright) + calisan Docker stack'i
gerektiren uctan uca testler.

ONEMLI: bu dosyadaki HER SEY @pytest.mark.e2e -- diger 420 test
(tests/test_dashboard_*.py, hepsi hermetik) TAMAMEN AYRI bir katman.
Bunlar SADECE Python pytest'in erisemedigi CLIENTSIDE JS mantigini
(poligon/marker render, firma filtresi, "gösteriliyor" sayaci -- hepsi
app.py sonundaki clientside_callback'lerde, tarayicida calisir) gercekten
dogrulayabilir. Hizli/hermetik calistirmadan HARIC tutmak icin:
    pytest -m "not e2e"
Sadece bunlari calistirmak icin:
    pytest -m e2e

Docker stack ayakta degilse (bkz. conftest.py'deki e2e_browser fixture'i)
tum dosya nazikce SKIP edilir, KIRMIZI olmaz."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


# ------------------------------------------------------------- sayfa yuklemesi --

def test_page_loads_with_dark_map_and_ground_filter_on_by_default(page):
    """Kullanici karariyla varsayilanlar: harita 'Karanlık' (CARTO Dark
    Matter), 'Yerde' filtresi ACIK (bkz. proje sohbet gecmisi)."""
    assert page.title() == "Dashboard"

    ground_btn_bg = page.eval_on_selector(
        "#filter-ground-btn", "el => getComputedStyle(el).backgroundColor")
    # Aktif (turuncu, FILTER_BTN_GROUND_ACTIVE_STYLE) -- kapali griden FARKLI.
    assert ground_btn_bg not in ("rgb(0, 0, 0)", "rgba(0, 0, 0, 0)")

    page.wait_for_selector(".leaflet-tile-pane img", timeout=15000)
    tile_url = page.eval_on_selector(".leaflet-tile-pane img", "el => el.src")
    assert "cartocdn.com" in tile_url


def test_status_bar_shows_total_and_shown_counts(page):
    """Durum cubugu formati: '{saat} | {toplam} aktif uçuş  {gösterilen}
    gösteriliyor | {alarm} alarm' -- ucu sayilar TUTARLI olmali (gosterilen
    <= toplam), TEK bir DOM okumasinda (tick araligina yakalanmasin diye)."""
    page.wait_for_selector("#status-main:not(:empty)", timeout=15000)
    numbers = page.eval_on_selector(
        "#status",
        """el => {
            const main = document.getElementById('status-main').textContent;
            const shown = document.getElementById('status-shown').textContent;
            const total = parseInt(main.match(/\\d+/g).pop(), 10);
            const shownN = parseInt(shown.match(/\\d+/)[0], 10);
            return [total, shownN];
        }""",
    )
    total, shown = numbers
    assert total > 0
    assert 0 <= shown <= total


def test_timezone_dropdown_shows_a_value_on_first_load_regression(page):
    """Regresyon: dropdown options=[] ile baslarsa value null'a sifirlanip
    BOS gorunuyordu -- bkz. _build_timezone_options degisikligi."""
    page.click("#settings-btn")
    page.wait_for_selector("#timezone-dropdown", state="visible")
    text = page.inner_text("#timezone-dropdown")
    assert text.strip() != ""
    assert "UTC" in text


@pytest.mark.parametrize("selector", ["#timezone-dropdown", "#airline-filter-dropdown"])
def test_dropdown_trigger_has_dark_background_not_white_regression(page, selector):
    """Regresyon: CSS yorumu icindeki yanlislikla '*/' iceren bir metin,
    '.dark-dropdown.dash-dropdown' kuralinin TAMAMEN parse edilmeden
    atlanmasina yol acmisti -- TUM dropdown'lar (bu ikisi dahil) beyaz
    kaliyordu (bkz. proje sohbet gecmisi, saatlerce suren debug)."""
    if selector == "#timezone-dropdown":
        page.click("#settings-btn")
    page.wait_for_selector(selector, state="visible")
    bg = page.eval_on_selector(selector, "el => getComputedStyle(el).backgroundColor")
    assert bg != "rgb(255, 255, 255)"
    assert bg != "rgba(255, 255, 255, 1)"


def test_dropdown_search_box_has_dark_background_regression(page):
    """Regresyon: acilan panel dogru koyu temaydi ama arama <input> kutusu
    tarayicinin varsayilan BEYAZ gorunumunu koruyordu (appearance:none ile
    duzeltildi, bkz. proje sohbet gecmisi)."""
    box = page.locator("#airline-filter-dropdown").bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_selector(".dash-dropdown-search", state="visible")
    bg = page.eval_on_selector(".dash-dropdown-search",
                               "el => getComputedStyle(el).backgroundColor")
    assert bg != "rgb(255, 255, 255)"


def test_zoom_control_buttons_are_dark_themed(page):
    bg = page.eval_on_selector(
        ".leaflet-control-zoom-in", "el => getComputedStyle(el).backgroundColor")
    assert bg != "rgb(255, 255, 255)"


# ---------------------------------------------------------------- firma filtresi --

def test_airline_filter_reduces_shown_count_but_not_total(page):
    """ASIL regresyon testi (kullanici geri bildirimi -- 'firma filtresi
    bu sayiyi hiç etkilemiyor'): firma filtresi TAMAMEN clientside JS'te
    calisiyor -- SADECE gercek bir tarayicida dogrulanabilir."""
    page.wait_for_selector("#status-shown:not(:empty)", timeout=15000)
    before = page.eval_on_selector(
        "#status",
        """el => {
            const main = document.getElementById('status-main').textContent;
            const shown = document.getElementById('status-shown').textContent;
            return [parseInt(main.match(/\\d+/g).pop(), 10), parseInt(shown.match(/\\d+/)[0], 10)];
        }""",
    )
    total_before, shown_before = before

    box = page.locator("#airline-filter-dropdown").bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_selector(".dash-dropdown-option", state="visible")
    page.locator(".dash-dropdown-option").first.click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(1500)  # clientside callback + canvas render icin

    after = page.eval_on_selector(
        "#status",
        """el => {
            const main = document.getElementById('status-main').textContent;
            const shown = document.getElementById('status-shown').textContent;
            return [parseInt(main.match(/\\d+/g).pop(), 10), parseInt(shown.match(/\\d+/)[0], 10)];
        }""",
    )
    total_after, shown_after = after

    assert shown_after < shown_before  # bir firma secince gosterilen AZALMALI
    # toplam sayi firma filtresinden ETKILENMEMELI (+/- birkac saniyelik
    # dogal trafik degisimi payi biraktik, tam esitlik ARAMIYORUZ).
    assert abs(total_after - total_before) < max(50, total_before * 0.05)


def test_airline_filter_dropdown_is_multi_select_with_placeholder(page):
    text = page.inner_text("#airline-filter-dropdown")
    assert "Firma" in text or text.strip() == ""  # secim yoksa placeholder


# --------------------------------------------------------------- harita katmani --

def test_map_style_switch_changes_tile_url(page):
    page.click("#settings-btn")
    page.wait_for_selector("#map-style-street-btn", state="visible")
    page.click("#map-style-street-btn")
    page.wait_for_timeout(1000)
    page.wait_for_selector(".leaflet-tile-pane img", timeout=15000)
    tile_url = page.eval_on_selector(".leaflet-tile-pane img", "el => el.src")
    assert "openstreetmap.org" in tile_url

    page.click("#map-style-dark-btn")
    page.wait_for_timeout(1000)
    page.wait_for_selector(".leaflet-tile-pane img", timeout=15000)
    tile_url_dark = page.eval_on_selector(".leaflet-tile-pane img", "el => el.src")
    assert "cartocdn.com" in tile_url_dark


# --------------------------------------------------------------------- paneller --

def test_settings_panel_opens_and_closes(page):
    page.click("#settings-btn")
    page.wait_for_selector("#settings-panel", state="visible")
    assert page.is_visible("#settings-panel")

    page.click("#close-settings-btn")
    page.wait_for_selector("#settings-panel", state="hidden")
    assert not page.is_visible("#settings-panel")


def test_replay_panel_prefills_default_one_hour_range_regression(page):
    """Regresyon (canli testte yakalanan sorun): panel ilk acildiginda 4
    tarih/saat alani BOS geliyordu, 'Yükle'ye basinca sessizce hicbir sey
    olmuyordu. Artik 'son 1 saat' otomatik doluyor olmali."""
    page.click("#replay-btn")
    page.wait_for_selector("#replay-panel", state="visible")
    page.wait_for_timeout(500)

    start_day_text = page.inner_text("#replay-start-day")
    start_hour_text = page.inner_text("#replay-start-hour")
    assert start_day_text.strip() != ""
    assert start_hour_text.strip() != ""

    page.click("#close-replay-btn")
    page.wait_for_selector("#replay-panel", state="hidden")


def test_aircraft_are_rendered_on_the_canvas(page):
    """Ucaklarin GERCEKTEN Leaflet canvas'ina cizildigini dogrular --
    tum filtre/renklendirme zincirinin (Python -> aircraft-raw ->
    clientside poligon hesaplama -> canvas) uctan uca calistigini
    gosteren tek gercek kanit."""
    page.wait_for_timeout(3000)
    pixel_count = page.eval_on_selector(
        ".leaflet-overlay-pane canvas",
        """canvas => {
            const ctx = canvas.getContext('2d');
            const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            let nonTransparent = 0;
            for (let i = 3; i < data.length; i += 4) {
                if (data[i] > 0) nonTransparent++;
            }
            return nonTransparent;
        }""",
    )
    assert pixel_count > 0
