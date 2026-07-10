"""
setup_local_windows.py
Docker'siz, native Windows kurulumu. Iki servis gerekiyor:
  1. Memurai  -- Redis'in resmi Windows portu (winget ile kurulur, otomatik
                 Windows servisi olarak calisir, script'in yonetmesine gerek yok)
  2. InfluxDB -- native Windows zip, script indirir/acar/baslatir

Bu script'i CALISMA KLASORUNDE (proje klasorunde) calistir -- influxdb/
alt klasoru ve influx_token.txt orada olusacak, ileride hep ayni klasorden
calistirmak gerekiyor (token ve veri orada kalici).

Kullanim:
    python setup_local_windows.py
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

INFLUXDB_VERSION = "2.9.1"
INFLUXDB_ZIP_URL = f"https://dl.influxdata.com/influxdb/releases/influxdb2-{INFLUXDB_VERSION}-windows_amd64.zip"

BASE_DIR = Path.cwd()
INFLUX_DIR = BASE_DIR / "influxdb"
TOKEN_FILE = BASE_DIR / "influx_token.txt"

INFLUX_ORG = "iha-org"
INFLUX_BUCKET = "adsb-history"
INFLUX_RETENTION_SECONDS = 7 * 24 * 3600  # 1 hafta
INFLUX_USER = "admin"
INFLUX_PASS = "admin12345"


def port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def check_memurai():
    print("=== 1) Redis (port 6379) kontrolu ===")
    if port_open("localhost", 6379):
        print("Redis zaten calisiyor (port 6379 acik).")
        return

    print("Redis calismiyor.")
    print()
    print("NOT: MSI tabanli kurulumlar (Memurai vb.) gercek admin sifresi")
    print("istiyorsa, onlari atlayip kurulum gerektirmeyen portable surumu")
    print("kullan:")
    print()
    print("  1. Tarayicidan indir (ZIP, MSI DEGIL):")
    print("     https://github.com/tporadowski/redis/releases")
    print("  2. C:\\redis klasorune cikar")
    print("  3. Terminalde:")
    print("       cd C:\\redis")
    print("       .\\redis-server.exe .\\redis.windows.conf")
    print("     (bu pencereyi acik birak)")
    print()
    print("Redis calisir hale gelince bu script'i tekrar calistir.")
    sys.exit(1)


def download_with_useragent(url: str, dest: Path):
    """dl.influxdata.com Python'un varsayilan urllib User-Agent'ini bot
    trafigi sayip 403 donduruyor -- tarayici gibi gorunen bir baslikla
    istek atiyoruz."""
    req = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36")
    })
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
        print()


def download_influxdb():
    if INFLUX_DIR.exists() and any(INFLUX_DIR.glob("influxd.exe")):
        print(f"InfluxDB zaten mevcut: {INFLUX_DIR}")
        return

    zip_path = BASE_DIR / f"influxdb2-{INFLUXDB_VERSION}.zip"
    if not zip_path.exists():
        print(f"Indiriliyor: {INFLUXDB_ZIP_URL}")
        download_with_useragent(INFLUXDB_ZIP_URL, zip_path)
        print(f"Indirildi: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print("Aciliyor...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(BASE_DIR / "_influx_extract")

    # zip icinde "influxdb2-2.9.1-windows_amd64" gibi bir klasor olusur,
    # onu sabit isimli influxdb/ klasorune tasi
    extracted = list((BASE_DIR / "_influx_extract").glob("influxdb2-*"))
    if extracted:
        extracted[0].rename(INFLUX_DIR)
    else:
        # bazi surumler dogrudan kok dizine acabiliyor
        (BASE_DIR / "_influx_extract").rename(INFLUX_DIR)

    print(f"Hazir: {INFLUX_DIR}")


def start_influxd():
    print("\n=== 2) InfluxDB baslatiliyor ===")
    if port_open("localhost", 8086):
        print("InfluxDB zaten calisiyor (port 8086 acik).")
        return

    influxd_exe = INFLUX_DIR / "influxd.exe"
    if not influxd_exe.exists():
        print(f"HATA: {influxd_exe} bulunamadi.")
        sys.exit(1)

    print("InfluxDB kendi penceresinde aciliyor (bu pencereyi KAPATMA,")
    print("arka planda calismaya devam etmesi lazim)...")

    subprocess.Popen(
        [str(influxd_exe)],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        cwd=str(INFLUX_DIR),
    )

    print("Port 8086 acilana kadar bekleniyor...")
    for _ in range(30):
        if port_open("localhost", 8086):
            print("InfluxDB hazir.")
            return
        time.sleep(2)

    print("HATA: InfluxDB 60 sn icinde acilmadi. Yeni acilan pencerede hata var mi kontrol et.")
    sys.exit(1)


def setup_influxdb():
    print("\n=== 3) InfluxDB kurulumu (org / bucket / token) ===")
    host = "http://localhost:8086"

    # Onceden kaydedilmis token varsa dogrula
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        req = urllib.request.Request(
            f"{host}/api/v2/buckets",
            headers={"Authorization": f"Token {token}"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            print(f"Kayitli token gecerli: {token[:20]}...")
            return token
        except Exception:
            print("Kayitli token artik gecersiz, yeniden kuruluyor...")

    # Setup daha once yapilmis mi kontrol et
    with urllib.request.urlopen(f"{host}/api/v2/setup", timeout=5) as r:
        setup_status = json.loads(r.read())

    if not setup_status.get("allowed", False):
        print("HATA: InfluxDB zaten kurulu ama token dosyasi kayip.")
        print("2.9.1'de tokenlar diskte hashli tutuluyor, eski token'i")
        print("kurtarmak guvenilir degil. Sifirlamak icin:")
        print(f"  1. InfluxDB penceresini kapat (Ctrl+C)")
        print(f"  2. Sil: {Path.home() / '.influxdbv2'}")
        print(f"  3. Bu script'i tekrar calistir")
        sys.exit(1)

    payload = json.dumps({
        "username": INFLUX_USER,
        "password": INFLUX_PASS,
        "org": INFLUX_ORG,
        "bucket": INFLUX_BUCKET,
        "retentionPeriodSeconds": INFLUX_RETENTION_SECONDS,
    }).encode()

    req = urllib.request.Request(
        f"{host}/api/v2/setup", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        result = json.loads(r.read())

    token = result["auth"]["token"]
    TOKEN_FILE.write_text(token)
    print(f"Kurulum tamam. Token kaydedildi: {TOKEN_FILE}")
    print(f"  Org: {INFLUX_ORG}  |  Bucket: {INFLUX_BUCKET}  |  Saklama: 7 gun")
    return token


def main():
    check_memurai()
    download_influxdb()
    start_influxd()
    token = setup_influxdb()

    print("\n" + "=" * 50)
    print("HAZIR. Simdi sirayla (ayri terminallerde):")
    print("  1. python adsb_poller.py     (veri toplamaya baslar)")
    print("  2. python app.py             (dashboard'u acar)")
    print("=" * 50)


if __name__ == "__main__":
    main()
