"""
setup_kafka_windows.py
Docker'siz, native Windows Kafka kurulumu. Zookeeper modu kullanilir
(KRaft modu Windows'ta bilinen dosya kilidi sorunlarina sahip -- Zookeeper
modu daha kanitlanmis, biz de Colab'da bunu kullanmistik).

Java (JRE) gerekiyor -- yoksa script sana kurulum komutunu verir.

Kullanim:
    python setup_kafka_windows.py
"""
import glob
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

KAFKA_VERSION = "3.9.2"
KAFKA_SCALA = "2.13"
KAFKA_URL = f"https://dlcdn.apache.org/kafka/{KAFKA_VERSION}/kafka_{KAFKA_SCALA}-{KAFKA_VERSION}.tgz"

BASE_DIR = Path.cwd()
# ONEMLI: Kafka'nin .bat script'leri classpath'i her jar'in TAM YOLUYLA
# birlestiriyor. Uzun proje yollarinda (C:\Users\...\Desktop\Dashboard\kafka)
# bu, Windows'un komut satiri karakter sinirini (8191) asip
# "The input line is too long" hatasi veriyor. Cozum: sabit, kisa bir yol.
KAFKA_DIR = Path("C:/kafka")

TOPICS = [
    ("adsb.flights", 3),  # ham parse edilmis ucus verisi (producer -> burasi)
    ("adsb.alerts", 1),   # model ekibi ileride buraya anomali alert'i yazacak
]

_JAVA_HOME = None  # find_java17() tarafindan doldurulur


def tail_log(path: Path, n: int = 30):
    if not path.exists():
        print(f"  (log dosyasi bulunamadi: {path})")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"  --- {path.name} son {min(n, len(lines))} satir ---")
    for line in lines[-n:]:
        print(f"  {line}")


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


def find_java17():
    """Java 17'yi PATH'e guvenmeden, bilinen kurulum klasorlerinden arar.
    Eski Java 8 kurulumlari sistem PATH'ine 'Oracle\\Java\\javapath' gibi
    kalici bir giris ekliyor -- kullanici PATH degisiklikleri bunu asamiyor.
    Bu yuzden 'java' komutuna guvenmek yerine dogru java.exe'yi dogrudan
    buluyoruz ve sadece kendi baslattigimiz process'lere veriyoruz."""
    search_bases = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Eclipse Adoptium",
        Path("C:/Program Files/Eclipse Adoptium"),
        Path("C:/Program Files (x86)/Eclipse Adoptium"),
    ]
    candidates = []
    for base in search_bases:
        if not base.exists():
            continue
        # hem jdk-17* hem jre-17* klasor adlarini kabul et
        candidates += list(base.glob("jdk-17*"))
        candidates += list(base.glob("jre-17*"))

    for c in candidates:
        java_exe = c / "bin" / "java.exe"
        if java_exe.exists():
            return c

    return None


def build_env(java_home: Path) -> dict:
    """Alt process'lere verilecek, JAVA_HOME'u dogru sekilde onceleyen
    ortam degiskenleri. Terminaldeki bozuk PATH'e guvenmiyoruz."""
    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    # "Turkish locale bug": Turkce Windows'ta String.toUpperCase() kucuk 'i'
    # harfini noktali buyuk 'I' (U+0130) yapiyor, normal 'I' degil. Kafka'nin
    # ic kodu bir config degerini (orn. "classic") buyuk harfe cevirip enum
    # ile eslestirirken bu yuzden patliyor (CLASSIC != CLASSİC). JVM'i sadece
    # bu process'ler icin Ingilizce locale'e zorluyoruz, sistem locale'ine
    # dokunmuyoruz.
    existing_opts = env.get("KAFKA_OPTS", "")
    env["KAFKA_OPTS"] = f"-Duser.language=en -Duser.country=US {existing_opts}".strip()
    return env


def check_java():
    global _JAVA_HOME
    print("=== 1) Java kontrolu ===")

    java_home = find_java17()
    if java_home is None:
        print("Java 17 bulunamadi (jdk-17*/jre-17* klasoru aranan yerlerde yok).")
        print()
        print("Su adresten JRE 17 indirip kur (sadece bu kullanici icin secebilirsin,")
        print("izin istemez):")
        print("  https://adoptium.net/temurin/releases/?version=17&os=windows&arch=x64&package=jre")
        print()
        print("Kurulumdan sonra bu script'i tekrar calistir.")
        sys.exit(1)

    _JAVA_HOME = java_home
    r = subprocess.run([str(java_home / "bin" / "java.exe"), "-version"],
                       capture_output=True, text=True)
    version_line = (r.stderr or r.stdout).splitlines()[0]
    print(f"Java 17 bulundu: {version_line}")
    print(f"  JAVA_HOME = {java_home}")
    print("  (terminaldeki 'java -version' hala eski surumu gosterebilir,")
    print("   onemli degil -- Kafka/Zookeeper'i dogrudan bu Java ile baslatiyoruz)")


def download_kafka():
    print("\n=== 2) Kafka indirme ===")
    if KAFKA_DIR.exists() and (KAFKA_DIR / "bin" / "windows" / "kafka-server-start.bat").exists():
        print(f"Kafka zaten mevcut: {KAFKA_DIR}")
        return

    tgz_path = BASE_DIR / f"kafka_{KAFKA_SCALA}-{KAFKA_VERSION}.tgz"
    if not tgz_path.exists():
        print(f"Indiriliyor: {KAFKA_URL}")
        urllib.request.urlretrieve(KAFKA_URL, tgz_path)
        print(f"Indirildi ({tgz_path.stat().st_size / 1e6:.0f} MB)")

    print("Aciliyor...")
    with tarfile.open(tgz_path) as tf:
        tf.extractall(BASE_DIR)

    extracted = BASE_DIR / f"kafka_{KAFKA_SCALA}-{KAFKA_VERSION}"
    if extracted.exists():
        extracted.rename(KAFKA_DIR)

    print(f"Hazir: {KAFKA_DIR}")


def start_zookeeper():
    print("\n=== 3) Zookeeper baslatiliyor ===")
    if port_open("localhost", 2181):
        print("Zookeeper zaten calisiyor (port 2181).")
        return

    bat = KAFKA_DIR / "bin" / "windows" / "zookeeper-server-start.bat"
    cfg = KAFKA_DIR / "config" / "zookeeper.properties"

    print("Zookeeper arka planda baslatiliyor (log: C:\\kafka\\zookeeper.log)...")
    log_path = KAFKA_DIR / "zookeeper.log"
    log_file = open(log_path, "w", encoding="utf-8")
    subprocess.Popen([str(bat), str(cfg)],
                     cwd=str(KAFKA_DIR),
                     env=build_env(_JAVA_HOME),
                     stdout=log_file, stderr=subprocess.STDOUT,
                     creationflags=subprocess.CREATE_NO_WINDOW)

    for _ in range(20):
        if port_open("localhost", 2181):
            print("Zookeeper hazir.")
            return
        time.sleep(2)
    print("HATA: Zookeeper 40 sn icinde acilmadi.")
    tail_log(log_path)
    sys.exit(1)


def start_kafka():
    print("\n=== 4) Kafka broker baslatiliyor ===")
    if port_open("localhost", 9092):
        print("Kafka zaten calisiyor (port 9092).")
        return

    bat = KAFKA_DIR / "bin" / "windows" / "kafka-server-start.bat"
    cfg = KAFKA_DIR / "config" / "server.properties"

    print("Kafka broker arka planda baslatiliyor (log: C:\\kafka\\kafka.log)...")
    log_path = KAFKA_DIR / "kafka.log"
    log_file = open(log_path, "w", encoding="utf-8")
    subprocess.Popen([str(bat), str(cfg)],
                     cwd=str(KAFKA_DIR),
                     env=build_env(_JAVA_HOME),
                     stdout=log_file, stderr=subprocess.STDOUT,
                     creationflags=subprocess.CREATE_NO_WINDOW)

    for _ in range(30):
        if port_open("localhost", 9092):
            print("Kafka hazir.")
            return
        time.sleep(2)
    print("HATA: Kafka 60 sn icinde acilmadi.")
    tail_log(log_path)
    sys.exit(1)


def create_topics():
    print("\n=== 5) Topic'ler olusturuluyor ===")
    bat = KAFKA_DIR / "bin" / "windows" / "kafka-topics.bat"

    for topic, partitions in TOPICS:
        r = subprocess.run([
            str(bat), "--create", "--topic", topic,
            "--bootstrap-server", "localhost:9092",
            "--partitions", str(partitions),
            "--replication-factor", "1",
            "--if-not-exists",
        ], capture_output=True, text=True, cwd=str(KAFKA_DIR), env=build_env(_JAVA_HOME))
        status = "OK" if r.returncode == 0 else f"HATA: {r.stderr[:200]}"
        print(f"  {topic:20s} ({partitions} partition)  {status}")


def main():
    check_java()
    download_kafka()
    start_zookeeper()
    start_kafka()
    create_topics()

    print("\n" + "=" * 55)
    print("KAFKA HAZIR. Simdi sirayla (ayri terminallerde):")
    print("  1. python adsb_producer.py       (adsb.lol -> Kafka)")
    print("  2. python dashboard_consumer.py  (Kafka -> Redis + InfluxDB)")
    print("  3. python app.py                 (dashboard)")
    print()
    print("Ekip arkadaslarina: KAFKA_SCHEMA.md dosyasindaki topic")
    print("sozlesmesine gore kendi consumer/producer'larini yazabilirler,")
    print("bu dosyalara veya senin kodlarina dokunmalarina gerek yok.")
    print("=" * 55)


if __name__ == "__main__":
    main()
