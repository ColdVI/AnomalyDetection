"""Dashboard/app.py -- gomulu <style> blogundaki CSS'in yorum-govenligi
testleri.

ONEMLI: bu butun dosya, bu oturumda GERCEKTEN yasanan ve saatlerce süren
bir hatanin regresyon testidir -- bir Turkce aciklama yorumu
(".Select-*/.VirtualizedSelect*") YANLISLIKLA literal bir "*/" iceriyordu,
bu da CSS yorumunu ERKEN kapatip hemen ardindan gelen
".dark-dropdown.dash-dropdown {...}" kuralinin TAMAMEN parse edilmeden
atlanmasina yol acmisti -- sonuc: TUM dropdown'larin (saat dilimi, dil,
firma filtresi...) arka plani beyaz kaliyordu.

ONEMLI (test yontemi hakkinda -- bu testler ilk yazildiginda basarisiz
oldugu icin BURAYA not dusuyorum): naif bir "yorumlari kaldir, secici
metinde hala var mi" kontrolu bu hatayi YAKALAMAZ -- cunku metin
seviyesinde secici string'i ORTADAN KAYBOLMUYOR, sadece bir onceki
(erken kapanan) yorumdan sizan DUZYAZI ile ayni "secici" konumuna
KARISIYOR (gercek bir CSS ayristirici bunu GECERSIZ SECICI olarak
tamamen reddediyor). Bunu dogru yakalayan test, yorumlar kaldirildiktan
SONRA her "{" ONCESI metnin GERCEK bir CSS secicisi gibi gorunup
gorunmedigini (Turkce harfler/apostrof/nokta gibi duzyazi karakterleri
ICERMEDIGINI) kontrol eden test_no_prose_leaks_into_selector_position_regression'dir
-- once REKONSTRUKTE EDILMIS orijinal hatali metinle DOGRULANDI (hatayi
yakaladigini), SONRA gercek/duzeltilmis dosyayla dogrulandi (yanlis
alarm vermedigini)."""

from __future__ import annotations

import re

from Dashboard.codes import app as dashapp

# Gecerli bir CSS secici (class/id/element/pseudo/attribute/kombinator)
# icinde bulunabilecek karakterler -- Turkce harfler, apostrof, nokta
# (cumle sonu anlaminda), parantez ICI DUZYAZI bunun DISINDA kalir.
_VALID_SELECTOR_CHARS = re.compile(r'[A-Za-z0-9_\-.,:# \t\n\[\]=\"\'*>+~()%]+')


def _extract_style_block(index_string: str) -> str:
    match = re.search(r"<style>(.*?)</style>", index_string, re.DOTALL)
    assert match is not None, "index_string icinde <style> blogu bulunamadi"
    return match.group(1)


def _strip_css_comments(css: str) -> str:
    """Gercek bir CSS ayristiricinin yaptigi gibi -- ilk '/*'den SONRAKI
    ILK '*/' yorumu kapatir (ic ice yorum YOKTUR)."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _selector_position_texts(stripped_css: str) -> list[str]:
    """Yorumlari kaldirilmis CSS'te her '{' ONCESI metni (yani "secici
    konumu"ndaki metni) dondurur."""
    return [m.group(1).strip() for m in re.finditer(r"([^{}]+)\{", stripped_css)]


# Bu oturumda eklenen/duzeltilen dropdown/slider CSS kurallarindan bir
# kritik alt kume -- yorum hatasi TEKRAR olursa bunlardan biri (ya da
# hepsi) yorum-govdesine "yutulup" post-strip metinde kaybolur.
_EXPECTED_SELECTORS = [
    ".dark-dropdown.dash-dropdown",
    ".dash-dropdown-content",
    ".dash-dropdown-search",
    ".dash-dropdown-value-item",
    ".dash-dropdown-option",
    ".leaflet-control-zoom-in",
]

# Gercekte yasanmis, hatali yorum metninin BIREBIR REKONSTRUKSIYONU --
# ".Select-*/.VirtualizedSelect*" icindeki "*/" yorumu erken kapatiyordu.
_RECONSTRUCTED_BUGGY_CSS = """
/* dcc.Dropdown -- Dash 4.x KENDI bilesenini kullaniyor (Radix UI
   tabanli, sinif isimleri dash-dropdown-*, RangeSlider'daki AYNI
   surum degisikligi). ESKI
   .Select-*/.VirtualizedSelect* kurallari (react-select
   dönemınden kalma) gercek DOM'da hic eslesmiyordu, kaldirildi. */
.dark-dropdown.dash-dropdown {
    background-color: #161625 !important;
}
"""


def test_style_block_exists_and_is_nonempty():
    css = _extract_style_block(dashapp.app_dash.index_string)
    assert len(css) > 100


def test_style_block_brace_count_is_balanced():
    css = _extract_style_block(dashapp.app_dash.index_string)
    stripped = _strip_css_comments(css)
    assert stripped.count("{") == stripped.count("}")


def test_expected_critical_selectors_are_present_as_real_rules():
    css = _extract_style_block(dashapp.app_dash.index_string)
    stripped = _strip_css_comments(css)
    missing = [sel for sel in _EXPECTED_SELECTORS if sel not in stripped]
    assert not missing


def test_no_prose_leaks_into_selector_position_regression():
    """Asil regresyon testi -- bkz. modul docstring'i. Gercek/duzeltilmis
    dosyada HER '{' oncesi metin gecerli bir CSS secicisi gibi gorunmeli,
    icine sizmis Turkce duzyazi (apostrof, Turkce harfler, cumle noktalari
    disinda kalan karakterler) OLMAMALI."""
    css = _extract_style_block(dashapp.app_dash.index_string)
    stripped = _strip_css_comments(css)
    offenders = [
        sel for sel in _selector_position_texts(stripped)
        if sel and not _VALID_SELECTOR_CHARS.fullmatch(sel)
    ]
    assert not offenders, (
        f"Secici konumunda gecersiz/duzyazi-benzeri metin bulundu: {offenders!r} -- "
        "bu, bir yorumun erken kapanip sonraki kuralin govdesine sizdigi "
        "anlamina gelebilir (bkz. modul docstring'i)."
    )


def test_regression_check_actually_catches_the_original_bug():
    """Test yonteminin GERCEKTEN ise yaradigini kanitlar -- rekonstrukte
    edilmis orijinal hatali metinle CALISTIRILDIGINDA bu kontrol hatayi
    YAKALAMALI (aksi halde yukaridaki 'gecti' sonucu yanlis guven verir)."""
    stripped = _strip_css_comments(_RECONSTRUCTED_BUGGY_CSS)
    offenders = [
        sel for sel in _selector_position_texts(stripped)
        if sel and not _VALID_SELECTOR_CHARS.fullmatch(sel)
    ]
    assert offenders, "Rekonstrukte edilmis hatali CSS, beklenmedik sekilde 'temiz' cikti"
