"""
adsb_producer.py
SADECE veri kaynagindan (adsb.lol, OpenSky, ...) veri ceker ve Kafka'ya
yazar. UCUS VERISI icin Redis/InfluxDB/MinIO gibi hicbir depoyu bilmez --
o is tuketicilerin (consumer) sorumlulugunda. Bu ayrim sayesinde ekip
arkadaslari kendi consumer'larini bu script'e hic dokunmadan yazabilir.

ONEMLI (istisna, kullanici istegi uzerine eklendi): producer artik TEK bir
KONTROL bilgisi icin Redis'e bakiyor -- "iha:settings:data_source" key'i,
dashboard'daki kaynak secim butonlarindan yazilan istenen kaynagi
(adsblol/opensky) tasir. Bu, YUKARIDAKI ayrimi BOZMUYOR -- hala hicbir
UCUS VERISI Redis'e yazilmiyor/okunmuyor, sadece "hangi kaynagi kullan"
seklinde tek bir hafif ayar okunuyor. Redis erisilemezse (baglanti yok/
cökmüş) producer SESSIZCE eski davranisina (CLI/ortam degiskeninden
sabit kaynak) doner -- Redis bu script icin ZORUNLU bir bagimlilik
DEGIL, sadece varsa kullanilan opsiyonel bir kontrol kanali.

============================================================
VERI KAYNAGI ADAPTORLERI -- yeni bir kaynak eklemek KOLAY:
  1) "fetch_all_<isim>(args) -> list[dict]" seklinde bir fonksiyon yaz.
     Donen HER dict, ASAGIDAKI ORTAK SEMAYA uymali (bkz. _normalize_common):
       icao24, callsign, lat, lon, alt, velocity, track, vertical_rate,
       category, squawk, emergency, is_military, is_ground, source, ts
     Kaynagin desteklemedigi alanlar icin makul varsayilan kullan:
     sayisal alanlar None, metin alanlari "", is_military False,
     is_ground False, emergency "none". _normalize_common() bunu senin
     yerine yapar, sadece elindeki degerleri ona ver.
  2) Dosyanin sonundaki SOURCES sozlugune "isim": {"fetch": fetch_all_<isim>,
     "interval": <varsayilan_saniye>} ekle. Kaynak, kimlik dogrulama basarili
     olunca vb. daha kisa bir araliga gecebiliyorsa "interval_override":
     (fetch'ten SONRA cagrilan, ya bir saniye sayisi ya da None donen bir
     fonksiyon) da ekleyebilirsin -- bkz. asagidaki "opensky" girdisi.
  3) --source <isim> (veya DATA_SOURCE ortam degiskeni) ile sec.
main() dongusu SOURCES disinda hicbir yerde kaynaga OZGU kod icermez --
hangi kaynak secilirse secilsin ayni sekilde calisir.
============================================================

YAPILANDIRMA -- KOMUT SATIRI *VEYA* ORTAM DEGISKENI (ikisi de calisir,
CLI verilirse o kazanir): bu, script'in NASIL baslatildigindan bagimsiz
calismasi icin -- bugun start_all.bat (Windows "set" = ortam degiskeni),
yarin Docker (docker run -e / docker-compose "environment:") -- AYNI
DATA_SOURCE/INTERVAL/... isimleri, hic kod degisikligi gerekmeden calisir.

  DATA_SOURCE   (--source)    varsayilan: adsblol -- Redis erisilebilirse
      VE dashboard'dan degistirilmisse, BU deger degil Redis'teki
      istenen kaynak kullanilir (bkz. REDIS_SOURCE_KEY).
  INTERVAL      (--interval)  varsayilan: 15 -- Redis'ten canli kaynak
      degistirilebilir durumdayken SOURCE_INTERVALS (adsblol=60,
      opensky=300) bunu GECERSIZ KILAR.
  LAT           (--lat)       varsayilan: 39.0   (sadece adsblol)
  LON           (--lon)       varsayilan: 35.0   (sadece adsblol)
  RADIUS        (--radius)    varsayilan: 12000  (sadece adsblol)
  KAFKA_BOOTSTRAP              varsayilan: localhost:9092 -- ONEMLI:
      Docker'da "localhost" container'in KENDI icini isaret eder, Kafka
      container'ina degil -- o zaman bunu servis adina (orn. "kafka:9092")
      ayarlaman GEREKECEK.
  REDIS_HOST / REDIS_PORT      varsayilan: localhost / 6379 -- canli
      kaynak degistirme icin (bkz. yukaridaki modul docstring'i notu).
      Erisilemezse producer bunu SESSIZCE yok sayar, DATA_SOURCE/INTERVAL
      ile calismaya devam eder.

Kullanim (CLI):
    python adsb_producer.py [--source adsblol] [--interval 15]
    python adsb_producer.py --source opensky --interval 300

Kullanim (ortam degiskeni -- Docker'da boyle kullanilacak):
    set DATA_SOURCE=opensky & set INTERVAL=300 & python adsb_producer.py
    # Docker: docker run -e DATA_SOURCE=opensky -e INTERVAL=300 ...
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import redis
import requests
from confluent_kafka import Producer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "adsb.flights"

DEFAULT_WORLD_RADIUS_NM = 12000

# Kaynak dashboard'dan CANLI degistirilebiliyor (bkz. modul docstring'i) --
# her kaynagin kendi hedef-aralik SANIYESI var (bkz. SOURCES sozlugu, dosya
# sonu), kullanicinin istegiyle SABIT: adsb.lol 60sn'de bir, OpenSky
# 300sn'de bir (anonim erisimde GUNDE SADECE 400 kredi -- kisa aralik
# kotayi dakikalar icinde tuketir). Bu, args.interval'i (CLI/ENV) Redis
# uzerinden kaynak degistirilebilir durumdayken GECERSIZ KILAR -- Redis
# yoksa/erisilemezse args.interval aynen kullanilmaya devam eder (eski/
# statik davranis).
# OPENSKY_CLIENT_ID/SECRET ile kimlik dogrulamali erisimde gunluk kredi
# 400 -> 4000'e cikiyor (bkz. _get_opensky_token()). CANLI OLCUM (bkz.
# "X-Rate-Limit-Remaining" loglari): dunya capinda bbox'siz TEK istek
# 1 DEGIL, 4 KREDI tuketiyor -- ilk denemede 60sn (1440 istek/gun x 4 =
# 5760 kredi) BUTCEYI ASIYORDU, gun ortasinda kota biterdi. Doğru hesap:
# 4000 kredi / 4 kredi-istek = 1000 istek/gun MAKSIMUM -> 86400/1000=86.4sn
# minimum aralik. 90sn'ye yuvarlayip guvenlik payi birakildi (960 istek/gun
# x 4 = 3840 kredi, ~%4 pay). Kotada fazladan payin oldugunu gorursen
# (loglardaki "kalan gunluk kredi" satirindan) dusurulebilir.
OPENSKY_AUTH_INTERVAL = 90
REDIS_SOURCE_KEY = "iha:settings:data_source"
REDIS_STATUS_KEY = "iha:producer_status"


def _normalize_common(icao24, callsign, lat, lon, alt_m, velocity_ms, track_deg,
                       vertical_rate_ms, category, squawk, emergency,
                       is_military, source, is_ground=False, signal_age_sec=None):
    """Tum kaynaklarin USTUNDE birlestigi ortak sema. Her adaptor kendi
    ham verisini normalize etmek icin bunu cagirir -- main() ve
    downstream (Kafka -> consumer -> dashboard) sadece bu ciktiyi gorur,
    hangi kaynaktan geldigini bilmesi gerekmez.

    signal_age_sec: bu KAYDIN ("uçak hâlâ dashboard'da görünüyor" ile
    "sinyali gerçekten ne kadar tazeydi" arasindaki fark) icin onemli --
    adsb.lol/readsb, bir ucaktan 60 saniyeye kadar mesaj gelmese bile
    onu listede TUTUYOR (kendi "seen" alaniyla bunu belirtiyor). Biz bu
    alani KULLANMADIGIMIZ icin, sinyali onlarca saniyedir kesilmis bir
    ucak bile bizim gozumuzde "taze" gorunuyordu -- haritada donuk/olu
    bir sinyal gibi kalabiliyordu. None ise kaynak bu bilgiyi
    saglamiyor demektir (guvenli varsayilan: taze kabul et)."""
    return {
        "icao24": icao24,
        "callsign": callsign,
        "lat": lat,
        "lon": lon,
        "alt": alt_m,
        "velocity": velocity_ms,
        "track": track_deg,
        "vertical_rate": vertical_rate_ms,
        "category": category,
        "squawk": squawk,
        "emergency": emergency,
        "is_military": is_military,
        "is_ground": is_ground,
        "source": source,
        "signal_age_sec": signal_age_sec,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================ adsb.lol --
#
# === HIZ ARASTIRMASI GECMISI (onemli, tekrar denenmesin) ===
# Once TEK bu dev 12000nm sorgu denendi -- fetch ~57-61s, produce ~0.1-0.2s
# (Kafka darbogaz DEGIL). 15-30sn hedefiyle, dunyayi 8 kucuk bolgeye bolup
# PARALEL cekmek denendi (Turkiye'nin 500nm sorgusu hizli oldugu icin
# "sure yaricaptan degil yogunluktan etkileniyor" varsayimiyla). Iki
# bulgu cikti: (1) adsb.lol ayni IP'den ESZAMANLI istegi sinirliyor --
# 8 istegi ayni anda atinca sadece 1-2'si gercek veri donuyordu, gerisi
# suphelice hizli "0 ucak" donuyordu. (2) --workers 1 (tam SIRALI) ile
# TUM bolgeler dogru veri donduruyordu ama toplam sure ~154-166 saniyeye
# cikti (Avrupa/Afrika/Rusya gibi yogun bolgelerin HER BIRI tek basina
# 37-44sn suruyor). SONUC: bolgesel bolme YAVASLATTI -- sorgu suresi
# donen KAYIT SAYISINDAN (yogunluktan) etkileniyor, 8 parcaya bolmek tek
# dev sorguda birlestirmekten daha pahaliya geliyor + paralellik yasak.
# Tek dev sorguya donuldu, 15-30sn hedefi bu API'yle ULASILAMAZ -- bu
# sunucu tarafinda bir sinir, istemci tarafinda asilamiyor.
# === /HIZ ARASTIRMASI GECMISI ===

def _fetch_adsblol_raw(lat, lon, radius_nm, max_retries=2, retry_delay=3):
    """adsb.lol'e HTTP GET. ONEMLI: buyuk (dunya capinda) yanitlarda
    bazen baglanti veri aktarimi TAMAMLANMADAN kesiliyor (ConnectionReset
    veya IncompleteRead hatasi) -- bu, temiz bir red (403/429) DEGIL,
    aktarim SIRASINDA kopma. Sebebi kesin degil (adsb.lol sunucu tarafi
    gecici yuk/instabilite VEYA yerel agdaki bir guvenlik/proxy katmani
    uzun sureli/buyuk indirmeleri kesiyor olabilir -- ikisini buradan
    ayirt edemiyoruz). Boyle bir kopma olursa, KISA bir bekleme sonrasi
    birkac kez tekrar deniyoruz -- gecici bir aksaklıksa bu genelde
    yeterli oluyor, degilse (max_retries tukenirse) None donup normal
    hata mesaji ile bir sonraki cycle'a devam ediliyor (crash YOK)."""
    url = f"https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}"
    for attempt in range(max_retries + 1):
        try:
            # ONEMLI: 90sn'den 120sn'ye cikarildi -- tipik basarili sorgu
            # ~55-60sn suruyor ama sunucu bazen daha yavas yanit veriyor
            # (adsb.lol tarafinda gecici yuk/gerginlik oldugunda) --
            # 90sn bu durumlarda ERKEN pes edip GEREKSIZ yere retry'a
            # geciyordu. BEDELI: gercekten kopmus bir baglantiyi fark
            # etmemiz de o kadar uzuyor (3 deneme x 120sn + 2x3sn bekleme
            # = en kotu durumda ~366sn/6dk, eskiden ~276sn/4.6dk idi).
            # KOK NEDEN (sunucu tarafi yavaslik/gerginlik) bizim
            # kontrolumuzde degil -- bu sadece TOLERANSI ayarliyor,
            # sorunu COZMUYOR.
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                data = r.json()
                if data.get("ac"):
                    return data
            return None  # HTTP hatasi (403/429/vb.) -- tekrar denemenin faydasi yok
        except Exception as e:
            if attempt < max_retries:
                print(f"  [uyari] adsb.lol fetch hatasi (deneme {attempt + 1}/"
                      f"{max_retries + 1}, {retry_delay}sn sonra tekrar denenecek): {e}")
                time.sleep(retry_delay)
            else:
                print(f"  [uyari] adsb.lol fetch hatasi (son deneme, vazgeciliyor): {e}")
    return None


def _parse_adsblol_aircraft(ac: dict):
    lat = ac.get("lat")
    lon = ac.get("lon")
    alt_raw = ac.get("alt_baro", ac.get("alt_geom", 0))
    # ONEMLI (askeri/sivil filtresiyle AYNI desende, ayri bir eksen):
    # adsb.lol, yerdeki (park/taksi) bir ucak icin "alt_baro" yerine
    # DUZ METIN "ground" donduruyor -- eskiden bu durumda kaydi TAMAMEN
    # ATIYORDUK (bkz. Dashboard/HANDOFF_UPDATE_2026-07-07.md Bolum 5,
    # "adsb.lol'un 'total aircraft' sayisi bizden yuksek cikabilir" notu
    # -- kok neden buydu). Artik ATMIYORUZ -- is_ground=True ile isaretleyip
    # irtifayi 0 kabul ediyoruz, haritada gosterip gostermemek tamamen
    # kullanicinin "Yerde" filtre butonuna (bkz. app.py) birakiliyor.
    is_ground = (str(alt_raw).lower() == "ground")
    if not lat or not lon or (not is_ground and not alt_raw):
        return None
    alt = 0.0 if is_ground else alt_raw
    icao = ac.get("hex", "").strip().lower()
    if not icao:
        return None

    # ONEMLI: adsb.lol her mesajda "gs" (ground speed) alanini
    # gondermiyor -- bazen pozisyon guncellemesi gelir ama hiz verisi
    # o an mevcut degildir (zayif sinyal, MLAT-only fix vb). "gs" eksikse
    # None birakiyoruz -- tuketici bu deger None ise InfluxDB'ye hic
    # yazmiyor, gercek bir bosluk oluyor, sahte sifir degil.
    raw_gs = ac.get("gs")
    velocity = round(float(raw_gs) * 0.5144, 1) if raw_gs is not None else None

    raw_baro_rate = ac.get("baro_rate")
    vertical_rate = (round(float(raw_baro_rate) * 0.00508, 2)
                     if raw_baro_rate is not None else None)

    # ONEMLI: ayni sekilde "track" (yon) eksikse sahte 0 (kuzey) YAZMIYORUZ.
    raw_track = ac.get("track")
    track = float(raw_track) if raw_track is not None else None

    # ASKERI/SIVIL AYRIMI: adsb.lol, ADSBExchange/readsb ile ayni "dbFlags"
    # bit alanini kullaniyor -- 1. bit (dbFlags & 1) askeri ucak demek.
    try:
        is_military = bool(int(ac.get("dbFlags", 0) or 0) & 1)
    except (TypeError, ValueError):
        is_military = False

    # ONEMLI: readsb (adsb.lol'un altyapisi) IKI AYRI tazelik alani
    # tutuyor -- "seen" (bu ucaktan HERHANGI bir mesaj -- irtifa/hiz vb. --
    # ne zaman geldi) ve "seen_pos" (POZISYON ozel olarak ne zaman
    # guncellendi). Biz haritada POZISYON gosterdigimiz icin asil onemli
    # olan seen_pos -- bir ucak zayif sinyal bolgesinde olup irtifa/hiz
    # mesajlari gelmeye devam ederken (yani "seen" dusuk kalirken, ucak
    # HER cycle'da listede gorunmeye devam eder, bizim cycle-id
    # temizligimiz HICBIR ZAMAN tetiklenmez) pozisyonu DAKIKALARCA
    # guncellenmemis olabilir -- bu durumu SADECE seen_pos yakalar.
    # seen_pos yoksa (readsb her zaman doldurmuyor) seen'e dusuyoruz.
    signal_age = ac.get("seen_pos", ac.get("seen"))

    return _normalize_common(
        icao24=icao, callsign=(ac.get("flight") or "").strip(),
        lat=round(float(lat), 6), lon=round(float(lon), 6),
        alt_m=round(float(alt) * 0.3048, 1),
        velocity_ms=velocity, track_deg=track, vertical_rate_ms=vertical_rate,
        category=ac.get("category", ""), squawk=ac.get("squawk", ""),
        emergency=ac.get("emergency", "none"), is_military=is_military,
        source="adsblol", is_ground=is_ground,
        signal_age_sec=(float(signal_age) if signal_age is not None else None),
    )


def fetch_all_adsblol(args):
    """adsb.lol -- tek buyuk yaricapli (--radius, varsayilan 12000nm =
    tum dunya) sorgu. ~57-61sn suruyor, bkz. yukaridaki HIZ ARASTIRMASI."""
    raw = _fetch_adsblol_raw(args.lat, args.lon, args.radius)
    if not raw:
        return []
    out = []
    for ac in raw.get("ac", []):
        rec = _parse_adsblol_aircraft(ac)
        if rec:
            out.append(rec)
    return out


# ============================================================ OpenSky --
#
# ONEMLI KISITLAR (denenmeden once oku):
#   - Anonim kullanici gunde SADECE 400 "kredi" alir -- surekli 15-60sn'de
#     bir sorgulamaya YETMEZ, birkac dakika icinde tukenir ve 429 (Too
#     Many Requests) almaya baslarsin. --interval'i BUYUK tut (orn. 300+).
#   - Kimlik dogrulamali (OAuth2 client credentials) erisim gunluk krediyi
#     4000'e cikarir -- OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET ortam
#     degiskenleri (opensky-network.org hesap ayarlarindan) verilirse
#     asagidaki _get_opensky_token() bunu OTOMATIK kullanir, interval de
#     OPENSKY_AUTH_INTERVAL'e duser (bkz. main() dongusu). Credential
#     yoksa/gecersizse SESSIZCE anonim erisime (300sn) duser, hata vermez.
#   - is_military VE category alanlari OpenSky'de yok -- her zaman
#     sirasiyla False ve "" donuyor (adsb.lol'deki gibi zengin degil).
#   - Bu, DUNYA capinda TEK istekle calisir (bbox verilmezse tum aktif
#     ucaklari doner) -- adsb.lol'deki gibi yaricap/bolge kavramı yok.

OPENSKY_EMERGENCY_SQUAWKS = {"7500": "unlawful", "7600": "nordo", "7700": "general"}

OPENSKY_TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/"
                      "opensky-network/protocol/openid-connect/token")

# Token ~30dk gecerli -- cache'leyip suresi dolmadan (60sn guvenlik payiyla)
# ONCE yeniliyoruz, her cycle'da yeniden istemek gereksiz/yavas olurdu.
_opensky_token_cache = {"token": None, "expires_at": 0}


def _get_opensky_token():
    """OAuth2 client_credentials ile access_token alir/cache'ler.
    OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET yoksa VEYA istek basarisiz
    olursa None doner -- cagiran taraf (fetch_all_opensky) bunu SESSIZCE
    anonim erisime duserek ele alir, KeyError/exception FIRLATMAZ."""
    client_id = os.environ.get("OPENSKY_CLIENT_ID")
    client_secret = os.environ.get("OPENSKY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    now = time.time()
    if _opensky_token_cache["token"] and now < _opensky_token_cache["expires_at"]:
        return _opensky_token_cache["token"]

    try:
        resp = requests.post(OPENSKY_TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }, timeout=10)
        if resp.status_code != 200:
            print(f"  [uyari] OpenSky token alinamadi (HTTP {resp.status_code}) "
                  f"-- anonim erisime duseniyor.")
            _opensky_token_cache["token"] = None
            return None
        data = resp.json()
        token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        _opensky_token_cache["token"] = token
        _opensky_token_cache["expires_at"] = now + expires_in - 60
        return token
    except Exception as e:
        print(f"  [uyari] OpenSky token hatasi: {e} -- anonim erisime duseniyor.")
        _opensky_token_cache["token"] = None
        return None


def _parse_opensky_state(state):
    """OpenSky state vector -- DICT DEGIL, POZISYONEL DIZI. Index'ler
    resmi API dokumantasyonuna gore (openskynetwork.github.io/opensky-api):
    0=icao24 1=callsign 2=origin_country 3=time_position 4=last_contact
    5=longitude 6=latitude 7=baro_altitude 8=on_ground 9=velocity
    10=true_track 11=vertical_rate 12=sensors 13=geo_altitude 14=squawk
    15=spi 16=position_source [17=category, sadece extended=true ile]."""
    try:
        icao24 = (state[0] or "").strip().lower()
        callsign = (state[1] or "").strip()
        lon, lat = state[5], state[6]
        baro_alt = state[7]
        on_ground = state[8]
        velocity, track, vertical_rate = state[9], state[10], state[11]
        squawk = state[14] or ""
        last_contact = state[4]  # unix zaman damgasi -- en son mesaj ne zaman alindi
    except (IndexError, TypeError):
        return None

    if not icao24 or lat is None or lon is None:
        return None

    # ONEMLI: adsb.lol'deki "seen" (saniye once) alaninin OpenSky
    # karsiligi -- last_contact bir UNIX ZAMAN DAMGASI (mutlak), "kac
    # saniye once" degil. Simdiki zamandan cikararak ayni birime
    # (saniye once) ceviriyoruz -- boylece iki kaynak da AYNI
    # "signal_age_sec" alanini tutarli sekilde dolduruyor.
    signal_age = (time.time() - last_contact) if last_contact is not None else None

    return _normalize_common(
        icao24=icao24, callsign=callsign,
        lat=round(float(lat), 6), lon=round(float(lon), 6),
        alt_m=round(float(baro_alt), 1) if baro_alt is not None else 0.0,
        # ONEMLI: OpenSky JSON'unda bir deger tam sayiysa (orn. 0, 0.5
        # degil) Python bunu int olarak parse eder. adsb.lol tarafi bu
        # alanlari hep float(...) ile aciyordu, burada UNUTULMUSTU --
        # InfluxDB bir alanin tipini (int/float) ILK YAZIMDA sabitliyor,
        # sonra farkli tip gelince "field type conflict" hatasi veriyor.
        # None'u KORUYORUZ (sahte 0 yazmiyoruz), ama None degilse MUTLAKA
        # float'a ceviriyoruz.
        velocity_ms=float(velocity) if velocity is not None else None,
        track_deg=float(track) if track is not None else None,
        vertical_rate_ms=float(vertical_rate) if vertical_rate is not None else None,
        category="",  # OpenSky kategori kodlari adsb.lol'unkiyle eslesmiyor
        squawk=squawk, emergency=OPENSKY_EMERGENCY_SQUAWKS.get(squawk, "none"),
        is_military=False,  # OpenSky bu bilgiyi vermiyor
        source="opensky", is_ground=bool(on_ground),
        signal_age_sec=round(signal_age, 1) if signal_age is not None else None,
    )


def fetch_all_opensky(args):
    """OpenSky Network states/all -- OPENSKY_CLIENT_ID/SECRET varsa
    kimlik dogrulamali (Bearer token, gunluk 4000 kredi), yoksa anonim
    (gunluk 400 kredi) erisim. Rate limit'e takilirsan --interval'i
    (anonimde 300sn+) artir."""
    url = "https://opensky-network.org/api/states/all"
    headers = {}
    token = _get_opensky_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 429:
            print("  [uyari] OpenSky: rate limit asildi (429) -- "
                  "--interval'i artir (anonim: gunde 400 kredi, "
                  "kimlik dogrulamali: 4000 kredi)")
            return []
        if r.status_code != 200:
            print(f"  [uyari] OpenSky fetch hatasi: HTTP {r.status_code}")
            return []
        # OpenSky, kimlik dogrulamali isteklerde kalan gunluk kredi
        # sayisini bu header'da veriyor -- OPENSKY_AUTH_INTERVAL (60sn)
        # sabit bir tahmindi, bu sayede GERCEK durumu goruyoruz, dusuk
        # kalirsa interval'i elle artirmak gerekebilir.
        remaining = r.headers.get("X-Rate-Limit-Remaining")
        if token and remaining is not None:
            print(f"  [opensky] kalan gunluk kredi: {remaining}")
        data = r.json()
    except Exception as e:
        print(f"  [uyari] OpenSky fetch hatasi: {e}")
        return []

    out = []
    for state in (data.get("states") or []):
        rec = _parse_opensky_state(state)
        if rec:
            out.append(rec)
    return out


# ============================================================ Kayit --
# Yeni kaynak eklemek icin buraya BIR giris ekle -- main() SADECE bu
# sozlukten okur, baska hicbir yeri (butonlar dahil app.py'deki
# DATA_SOURCE_DEFS ayri bir kayittir, orada da ayni isimle eslesen bir
# girdi eklemen gerekir) degistirmen gerekmiyor.
#   fetch:              fetch_all_<isim>(args) -> list[dict]
#   interval:           varsayilan hedef aralik (saniye)
#   interval_override:  (opsiyonel) fetch()'ten SONRA cagrilir; bir sayi
#                       donerse "interval"i o cycle icin GECERSIZ KILAR,
#                       None donerse (orn. kimlik dogrulama basarisizsa)
#                       "interval" aynen kullanilmaya devam eder.
SOURCES = {
    "adsblol": {"fetch": fetch_all_adsblol, "interval": 60},
    "opensky": {
        "fetch": fetch_all_opensky,
        "interval": 300,
        # Kimlik dogrulamali erisimde (bkz. OPENSKY_AUTH_INTERVAL yorumu)
        # daha kisa araliga gec -- token cache'i fetch_all_opensky() SIRASINDA
        # dolar/dogrulanir, o yuzden bu SADECE fetch'ten sonra anlamli.
        "interval_override": lambda: (
            OPENSKY_AUTH_INTERVAL if _opensky_token_cache["token"] else None
        ),
    },
}


def delivery_report(err, msg):
    if err is not None:
        print(f"  [uyari] teslim hatasi: {err}")


def main():
    # ONEMLI: her arg once CLI'dan (--source vb.), verilmezse ortam
    # degiskeninden (DATA_SOURCE vb.), o da yoksa sabit varsayilandan
    # okunuyor. Boylece bu script'i HIC DEGISTIRMEDEN hem "python
    # adsb_producer.py --source opensky" ile hem de Docker'da
    # "docker run -e DATA_SOURCE=opensky ..." ile calistirabilirsin.
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES.keys()),
                    default=os.environ.get("DATA_SOURCE", "adsblol"),
                    help="Veri kaynagi (varsayilan: adsblol; ortam degiskeni: DATA_SOURCE)")
    ap.add_argument("--interval", type=int,
                    default=int(os.environ.get("INTERVAL", "15")),
                    help="Ortam degiskeni: INTERVAL")
    ap.add_argument("--lat", type=float, default=float(os.environ.get("LAT", "39.0")),
                    help="Sadece --source adsblol icin. Ortam degiskeni: LAT")
    ap.add_argument("--lon", type=float, default=float(os.environ.get("LON", "35.0")),
                    help="Sadece --source adsblol icin. Ortam degiskeni: LON")
    ap.add_argument("--radius", type=int,
                    default=int(os.environ.get("RADIUS", str(DEFAULT_WORLD_RADIUS_NM))),
                    help="Sadece --source adsblol icin. Ortam degiskeni: RADIUS")
    args = ap.parse_args()

    # ONEMLI: Redis ZORUNLU DEGIL -- sadece "hangi kaynagi kullan" kontrol
    # sinyali icin (bkz. modul docstring'i). Baglanti basarisiz olursa
    # rdb=None birakip SESSIZCE eski/statik davranisa (args.source sabit)
    # donuyoruz -- bu script'in Redis'siz de (orn. teammate'in kendi
    # ortaminda) calismaya devam etmesi gerekiyor.
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    try:
        rdb = redis.Redis(host=redis_host, port=redis_port, db=0,
                          decode_responses=True, protocol=2,
                          socket_connect_timeout=3)
        rdb.ping()
        print(f"Redis baglandi ({redis_host}:{redis_port}) -- kaynak "
              f"dashboard'dan canli degistirilebilir.")
    except Exception as e:
        rdb = None
        print(f"[uyari] Redis'e baglanilamadi ({e}) -- kaynak SABIT "
              f"kalacak (DATA_SOURCE={args.source}), dashboard'dan "
              f"degistirilemez.")

    print(f"Kafka producer hazir (topic={TOPIC})")
    print(f"Veri kaynagi (baslangic): {args.source}")
    print("NOT: gercek cycle suresi hedef araliktan uzun olabilir -- asagidaki")
    print("     [timing] satirindan fetch/produce'un ne kadar surdugunu izle.")
    print("Durdurmak icin Ctrl+C\n")

    # ONEMLI: linger.ms/batch.size/compression.type -- binlerce mesajlik
    # burst'leri daha az sayida, daha buyuk ag paketiyle gondermek icin.
    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 50,
        "batch.size": 262144,       # 256 KB (varsayilan 16 KB)
        "compression.type": "lz4",
        "queue.buffering.max.messages": 200000,
    })

    stats = {"cycles": 0, "total": 0}

    try:
        while True:
            t0 = time.time()

            # ONEMLI: kaynak/interval HER CYCLE'DA yeniden okunuyor (sabit
            # DEGIL) -- boylece dashboard'daki kaynak butonu bir onceki
            # cycle biterken basilmis olsa bile bir SONRAKI cycle bunu
            # yakalar (en kotu durumda bir cycle suresi -- 60/300sn --
            # gecikme olur, bkz. modul docstring'i). Redis erisilemezse
            # veya deger tanidik degilse args.source'a (CLI/ENV) duser.
            current_source = args.source
            if rdb is not None:
                try:
                    requested = rdb.get(REDIS_SOURCE_KEY)
                    if requested in SOURCES:
                        current_source = requested
                except Exception:
                    pass  # Redis gecici erisilemez -- bir onceki/varsayilan kaynakla devam
            entry = SOURCES[current_source]
            fetch_fn = entry["fetch"]
            current_interval = entry.get("interval", args.interval)

            t_fetch_start = time.time()
            records = fetch_fn(args)
            t_fetch = time.time() - t_fetch_start

            # ONEMLI: interval_override, fetch_fn(args) CAGRILDIKTAN SONRA
            # calisiyor -- opensky ornegindeki gibi, _get_opensky_token()
            # cache'i bu cagri SIRASINDA doldu/dogrulandi (credential
            # gecerliyse). Override None donerse (credential yok/gecersiz)
            # yukaridaki sabit "interval" degeri aynen kullanilmaya devam
            # eder -- gercekte dogrulanmamis bir kimlikle YANLISLIKLA
            # hizli/kota-asan bir araliga gecilmez.
            override_fn = entry.get("interval_override")
            if override_fn:
                overridden = override_fn()
                if overridden:
                    current_interval = overridden

            if records:
                # ONEMLI: bu producer UCUS VERISI icin Redis/TTL/pencere
                # kavramindan TAMAMEN habersiz (tam da "producer hicbir
                # depoyu bilmez" ilkesine uygun) -- SADECE burada eklenen
                # "cycle_id", tuketiciye (dashboard_consumer.py) "bu
                # kayitlar AYNI cycle'a ait" bilgisini tasiyor. Tuketici,
                # cycle_id DEGISTIGINDE "onceki cycle TAMAMLANDI" sinyalini
                # alip o cycle'da hic gorulmeyen eski kayitlari SILIYOR --
                # SANIYE TAHMINI (WINDOW_SEC gibi) YOK, sinir producer'in
                # KENDI dogal cycle sinirindan geliyor. Bu, eskiden
                # zaman-pencereli yaklasimin verdigi "birden fazla
                # cycle'in birlesimi gosteriliyor" fazlaligini (bkz.
                # proje sohbet gecmisi, %13'e kadar olcduk) SIFIRA
                # indiriyor -- her zaman TAM OLARAK bir onceki
                # TAMAMLANMIS cycle ile karsilastiriliyor.
                stats["cycles"] += 1
                t_produce_start = time.time()
                for rec in records:
                    rec["cycle_id"] = stats["cycles"]
                    producer.produce(
                        TOPIC, key=rec["icao24"],
                        value=json.dumps(rec).encode(),
                        callback=delivery_report,
                    )
                producer.flush()
                t_produce = time.time() - t_produce_start

                stats["total"] += len(records)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{len(records)} ucus -> Kafka (kaynak={current_source}, "
                      f"cycle={stats['cycles']}, toplam={stats['total']})")
                print(f"  [timing] fetch={t_fetch:.1f}s  produce={t_produce:.1f}s  "
                      f"cycle-toplam={time.time() - t0:.1f}s  "
                      f"(hedef araligi={current_interval}s)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] veri alinamadi "
                      f"(kaynak={current_source}, fetch={t_fetch:.1f}s)")

            # Dashboard'un "gercekten hangi kaynak/aralik aktif" gorebilmesi
            # icin geri bildirim -- istek (REDIS_SOURCE_KEY) ile GERCEKTE
            # aktif olan (bu key) FARKLI olabilir (henuz bir sonraki cycle'a
            # gecilmedi), UI bu ikisini KARSILASTIRIP "gecis bekleniyor"
            # gosterebiliyor. ex=900 -- producer cokerse eski durum sonsuza
            # kadar "aktif" gorunmesin diye kisa sureli TTL.
            if rdb is not None:
                try:
                    rdb.set(REDIS_STATUS_KEY, json.dumps({
                        "source": current_source,
                        "interval": current_interval,
                        "opensky_authenticated": bool(_opensky_token_cache["token"]),
                        "cycle_id": stats["cycles"],
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }), ex=900)
                except Exception:
                    pass

            sleep_time = max(0, current_interval - (time.time() - t0))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nDurduruldu. Toplam: {stats['cycles']} cycle, {stats['total']} kayit")


if __name__ == "__main__":
    main()
