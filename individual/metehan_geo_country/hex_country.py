"""hex_country.py -- ICAO 24-bit Mode S hex adresini ULKEYE cevirir.

Kaynak: rikgale/ICAOList reposu (data/ICAOHexRange.csv), ICAO Annex 10 hex
blok tahsis tablosunun acik-kaynak topluluk tarafindan derlenmis hali (aynen
tar1090/dump1090 gibi araclarin "bayrak" ozelliginin dayandigi kaynak).

Tablo YUVALANMIS (nested) araliklar iceriyor: bazi genis "(reserved, EUR/NAT)"
gibi kita bloklari, icinde daha dar ulke araliklarini barindiriyor (CSV'de
4. kolon "nested" olarak isaretli). Bir hex birden fazla araliga
duserse (genis rezerve blok + dar ulke bloku), EN DAR (en spesifik) araligi
sec -- bu dogru ulkeyi, yanlislikla genis "(reserved, ...)" etiketini degil.

ONEMLI SINIRLAMA: Bu tablo ucagin KAYITLI OLDUGU (tescil) ulkeyi verir, o an
hangi ulke uzerinde uctugunu DEGIL. Bir Turk tescilli ucak Almanya uzerinde
ucabilir; bu modul sadece "Turkiye" doner, "Almanya" degil.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RANGE_CSV = DATA_DIR / "ICAOHexRange.csv"

# Bu iki onek, gercek bir ulkeye degil ICAO'nun kendi rezervasyonuna veya
# hic tahsis edilmemis bosluga isaret eder -- coverage hesabinda ayri
# tutulmasi gerekir (bkz. step1_coverage.py).
NON_COUNTRY_PREFIXES = ("(unallocated)", "(reserved")


@dataclass(frozen=True)
class HexRange:
    start: int
    end: int
    country: str

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def is_country(self) -> bool:
        return not self.country.startswith(NON_COUNTRY_PREFIXES)


def load_ranges(path: Path = RANGE_CSV) -> list[HexRange]:
    # csv.reader (not naive str.split(",")) -- bazi ulke adlari tirnakli ve
    # ic virgul iceriyor (orn. "(reserved, AFI)"), naive split bunlari
    # kirar (2026-07-08'de 7 hex'in yanlislikla "country" kategorisine
    # sizdigi bulundu, bkz. step1_coverage.py verify).
    ranges: list[HexRange] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            start_s, end_s, country = row[0].strip(), row[1].strip(), row[2].strip()
            if not start_s or not end_s:
                continue
            try:
                start = int(start_s, 16)
                end = int(end_s, 16)
            except ValueError:
                continue
            ranges.append(HexRange(start=start, end=end, country=country))
    return ranges


class HexCountryLookup:
    """Sirali aralik listesi uzerinde en-spesifik-eslesme lookup'i.

    Sadece 200 satir var (200 aralik) -- 11 binlik benzersiz hex seti icin
    bile lineer tarama trivial hizli, ikili arama gibi bir optimizasyona
    gerek yok.
    """

    def __init__(self, ranges: list[HexRange] | None = None):
        self.ranges = ranges if ranges is not None else load_ranges()

    def lookup(self, hex_str: str) -> tuple[str | None, str]:
        """Donus: (ulke_adi_veya_None, kategori).

        kategori: "country" | "reserved" | "no_match"
        - "country": gercek bir ulkeye eslesti, ulke_adi doludur.
        - "reserved": ICAO rezerve/tahsis-edilmemis bir bloga dustu (ulke yok).
        - "no_match": tabloda hic karsiligi yok (tablo eksik veya hex gecersiz).
        """
        try:
            val = int(hex_str, 16)
        except (ValueError, TypeError):
            return None, "no_match"

        candidates = [r for r in self.ranges if r.start <= val <= r.end]
        if not candidates:
            return None, "no_match"

        # En dar (en spesifik) araligi sec -- nested ulke bloklari, onlari
        # saran genis "(reserved, ...)" bloklarindan HER ZAMAN daha dardir.
        best = min(candidates, key=lambda r: r.width)
        if best.is_country:
            return best.country, "country"
        return None, "reserved"
