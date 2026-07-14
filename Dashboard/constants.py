"""app.py'den cikarildi, adim 1 -- kucuk skaler config sabitleri, saf veri, yan
etki yok. UYARI: REDIS_HOST/INFLUX_HOST/INFLUX_TOKEN gibi ortam degiskeni
okuyan GERCEK ayarlar burada DEGIL -- onlar app.py'de kaliyor (SystemExit
firlatabilen yan etkileri var, bu dosyanin "saf veri" ilkesini bozar)."""

# Kullanicinin ayarlardan sectigi "bayat sinyal" esigi (saniye) -- bu
# esigin ALTINDAKI ucaklar tam opak, USTUNDEKILER soluk (bkz. update_map
# icindeki opacity hesabi). Onceden SABIT bir 10-40sn dogrusal soluklasma
# vardi -- adsb.lol (60sn'de bir sorgulama) icin makuldu ama OpenSky
# (90-300sn'de bir sorgulama, bkz. uav_producer.py SOURCES) icin neredeyse
# HER ucak daha ilk fetch'te esigin ustune cikip ekrandaki NEREDEYSE TUM
# filo soluk gorunuyordu -- kullanici geri bildirimi bu. Kaynaga gore SABIT
# bir esik yerine, kullanicinin secili kaynaga gore kendi esigini ayarlar
# panelinden secmesini sagliyoruz.
SIGNAL_STALENESS_OPTIONS = [30, 60, 120, 300, 600, 1800, 3600]  # saniye --
                            # 30sn/1dk/2dk/5dk/10dk/30dk/1sa
DEFAULT_SIGNAL_STALENESS_SEC = 60
STALE_SIGNAL_OPACITY = 0.35  # esigi asan ucaklarin gorunecegi soluk deger
                              # (0 degil -- tamamen kaybolmasin, hep bir iz kalsin)

# ============================================================ Kayit --
# Yeni bir kaynak eklemek icin buraya BIR giris ekle (ayrica
# uav_producer.py'deki SOURCES sozlugune ayni "key" ile fetch fonksiyonunu
# ekle) -- butonlar ve callback bu listeden turetiliyor, baska hicbir yeri
# degistirmen gerekmiyor.
DATA_SOURCE_DEFS = [
    {"key": "adsblol", "label": "adsb.lol"},
    {"key": "opensky", "label": "OpenSky"},
]
DATA_SOURCES = tuple(d["key"] for d in DATA_SOURCE_DEFS)
DEFAULT_DATA_SOURCE = DATA_SOURCE_DEFS[0]["key"]
REDIS_DATA_SOURCE_KEY = "iha:settings:data_source"
REDIS_PRODUCER_STATUS_KEY = "iha:producer_status"

REPLAY_MAX_RANGE_HOURS = 2  # sorgu/payload boyutunu makul tutmak icin sabit ust sinir

GEOCODE_CACHE_TTL = 60 * 60 * 24 * 30  # 30 gun -- yer adlari neredeyse hic degismez
GEOCODE_MAX_LOOKUPS_PER_REQUEST = 16  # guvenlik siniri, bkz. _reverse_geocode

DEFAULT_TIMEZONE = 3  # UTC+3, Turkiye -- dropdown "value" olarak int kullanilir

# Haritadaki rota izi (polyline) artik sabit saat degil, "son ucus" --
# gercek ucus/blok verisi olmadigi icin bir HEURISTIK: ardisik iki konum
# noktasi arasinda bu esikten (dakika) BUYUK bir bosluk, "onceki ucus
# bitti, yenisi basladi" sayilir (bkz. update_flight_path). Normal ucus
# icinde ardisik nokta araligi (~15-90sn, producer cycle suresine bagli)
# bu esigin COK altinda kalir.
FLIGHT_GAP_THRESHOLD_MIN = 20

# Gecmis grafigi tarih araligi secicisi icin saat secenekleri (0-23) --
# sabit, dile/tarihe bagli degil, gun secenekleri gibi callback'te
# yeniden hesaplanmasina gerek yok.
HISTORY_HOUR_OPTIONS = [{"label": f"{h:02d}", "value": h} for h in range(24)]

# Firma (havayolu) filtresi -- ICAO 3 harfli cagri kodu on-eki -> firma adi.
# KUCUK, ELLE KURATORLU bir liste (bu projede sik gorulen/taninan ~40
# havayolu) -- kapsamli bir ICAO veritabani DEGIL, kullanici tercihi boyle
# (bkz. proje sohbet gecmisi). Listede OLMAYAN cagri kodlari (askeri, genel
# havacilik, taninmayan/bolgesel havayollari) filtre BOSKEN her zaman
# gorunur kalir -- bir veya daha fazla firma SECILINCE haritada SADECE o
# firma(lar) gosterilir (bkz. asagidaki clientside_callback, filtreleme
# TARAYICIDA/JS'te yapiliyor -- update_map'e YENI bir Input EKLENMEDI,
# boylece bu oturumda iki kez duzeltilen "pahali sunucu isteği + yarış
# durumu" hatasi SINIFI bastan hic olusmuyor -- callsign zaten cekilen
# veride mevcut, ekstra ag gidis-donusune gerek yok).
AIRLINE_PREFIXES = {
    "THY": "Turkish Airlines", "PGT": "Pegasus", "SXS": "SunExpress",
    "RYR": "Ryanair", "EZY": "easyJet", "WZZ": "Wizz Air", "VLG": "Vueling",
    "DLH": "Lufthansa", "AFR": "Air France", "BAW": "British Airways",
    "KLM": "KLM", "IBE": "Iberia", "AZA": "ITA Airways", "SWR": "Swiss",
    "AUA": "Austrian Airlines", "LOT": "LOT Polish Airlines",
    "TAP": "TAP Air Portugal", "SAS": "SAS", "FIN": "Finnair",
    "ELY": "El Al", "AFL": "Aeroflot", "BEL": "Brussels Airlines",
    "UAE": "Emirates", "QTR": "Qatar Airways", "ETD": "Etihad Airways",
    "SVA": "Saudia", "MEA": "Middle East Airlines",
    "DAL": "Delta Air Lines", "AAL": "American Airlines",
    "UAL": "United Airlines", "ACA": "Air Canada", "JAL": "Japan Airlines",
    "ANA": "All Nippon Airways", "CPA": "Cathay Pacific",
    "SIA": "Singapore Airlines", "QFA": "Qantas", "KAL": "Korean Air",
    "THA": "Thai Airways", "GIA": "Garuda Indonesia",
    "CES": "China Eastern", "CSN": "China Southern", "CCA": "Air China",
    "ETH": "Ethiopian Airlines", "MSR": "EgyptAir",
    "RJA": "Royal Jordanian",
}
