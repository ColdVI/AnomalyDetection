"""Build an Overleaf-ready detailed no-go report from frozen pilot artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def _num(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _role_rows(result: dict[str, Any], role: str) -> list[str]:
    rows = []
    for contract_name in ("critical", "advisory"):
        for method in ("px4_native", "cusum", "lstm"):
            metrics = result["contracts"][contract_name]["methods"][method]
            rows.append(
                " & ".join(
                    [
                        role,
                        contract_name,
                        method.replace("_", r"\_"),
                        _pct(metrics["events"]["recall"]),
                        _pct(metrics["events"]["wilson_95"]["lower"]),
                        _pct(metrics["events"]["by_fault_mode"]["3"]["recall"]),
                        _pct(metrics["events"]["by_fault_mode"]["4"]["recall"]),
                        _num(metrics["burden"]["episodes_per_scoreable_flight_hour"]),
                        _pct(metrics["burden"]["scoreable_coverage"]),
                        "PASS" if metrics["passes_gate"] else "FAIL",
                    ]
                )
                + r" \\"
            )
    return rows


def _stress_rows(stress: dict[str, Any]) -> list[str]:
    rows = []
    for domain in ("SIL-Wind", "HIL-Wind"):
        for contract_name in ("critical", "advisory"):
            for method in ("px4_native", "cusum", "lstm"):
                metrics = stress["domains"][domain]["contracts"][contract_name]["methods"][
                    method
                ]
                rows.append(
                    " & ".join(
                        [
                            domain,
                            contract_name,
                            method.replace("_", r"\_"),
                            str(stress["domains"][domain]["n_cases"]),
                            _num(metrics["episodes_per_scoreable_flight_hour"]),
                            _pct(metrics["scoreable_coverage"]),
                        ]
                    )
                    + r" \\"
                )
    return rows


def build_report(artifact_dir: str | Path) -> Path:
    artifact_dir = Path(artifact_dir)
    preflight = _load(artifact_dir / "preflight.json")
    calibration = _load(artifact_dir / "calibration_result.json")
    development = _load(artifact_dir / "development_result.json")
    rehearsal = _load(artifact_dir / "rehearsal_result.json")
    stress = _load(artifact_dir / "wind_stress_result.json")
    role_counts = preflight["role_counts"]
    diagnostic = development["lstm_magnitude_diagnostic"]
    calibration_rows = []
    for contract_name in ("critical", "advisory"):
        contract = calibration["contracts"][contract_name]
        calibration_rows.append(
            " & ".join(
                [
                    contract_name,
                    _num(contract["cusum"]["selected"]["threshold"], 4),
                    _num(contract["cusum"]["selected"]["episodes_per_hour"]),
                    _num(contract["lstm"]["selected"]["total_alpha"], 6),
                    _num(contract["lstm"]["selected"]["episodes_per_hour"]),
                ]
            )
            + r" \\"
        )
    result_rows = _role_rows(development, "development") + _role_rows(
        rehearsal, "rehearsal"
    )
    stress_rows = _stress_rows(stress)
    output = artifact_dir / "uav_gnss_integrity_v1_final_no_go_report.tex"
    output.write_text(
        rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[turkish]{{babel}}
\usepackage{{geometry,booktabs,longtable,array,xcolor,hyperref}}
\geometry{{margin=2cm}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{UAV GNSS Bütünlük Fizibilitesi v1\\Nihai Development--Rehearsal NO-GO Raporu}}
\author{{AnomalyDetection Projesi}}
\date{{16 Temmuz 2026}}
\begin{{document}}
\maketitle

\section{{Yönetici Özeti}}
Bu ``last dance'' çalışması genel amaçlı anomaly detector iddiasını tekrar
denememiş; yalnız PX4 üzerinde mevcut telemetriyle GNSS bütünlüğü için dar,
ön-kayıtlı bir fizibilite pilotu yürütmüştür. Üç yöntem değerlendirilmiştir:
PX4-native innovation/test-ratio kuralları, çok-kanallı Page CUSUM ve
location/scale üreten contextual LSTM.

\textbf{{Karar: mevcut veri ve enstrümantasyonla ürün adayı elde edilememiştir.}}
Hiçbir yöntem hem development hem rehearsal kapılarını birlikte geçmemiştir.
Bu nedenle kör holdout açılmamış ve sonuçlara bakılarak eşik gevşetilmemiştir.
Durum \texttt{{NO-GO / not achievable with current data and instrumentation}}
olarak kaydedilmelidir.

\section{{Problem Sözleşmesi}}
Tek soru şudur: RflyMAD gerçek uçuşlarındaki GNSS noise ve scale-factor
arızaları, doğal normal uçuşta kabul edilebilir operatör yükü korunarak
zamanında tespit edilebilir mi?
\begin{{itemize}}
  \item Kritik kapı: 5 saniye içinde recall, en fazla 2 alarm episode/uçuş-saat.
  \item Advisory kapı: 15 saniye içinde recall, en fazla 12 alarm episode/uçuş-saat.
  \item Normal coverage alt sınırı \%95; fault-event evaluable coverage alt sınırı \%90.
  \item Satır, episode, event, uçuş ve scoreable-flight-hour birimleri ayrıdır.
  \item \texttt{{not\_evaluable}} normal veya anomaly sınıfına zorlanmamıştır.
\end{{itemize}}
Config SHA-256: \texttt{{{preflight["config_sha256"]}}}.

\section{{Veri Denetimi ve Rol Ayrımı}}
\begin{{tabular}}{{lr}}
\toprule Rol & Uçuş sayısı \\ \midrule
Fit & {role_counts["fit"]} \\
Calibration & {role_counts["calibration"]} \\
Development & {role_counts["development"]} \\
Rehearsal & {role_counts["rehearsal"]} \\
Mühürlü holdout & {role_counts["holdout"]} \\
Karantina & {role_counts["quarantine"]} \\
\bottomrule
\end{{tabular}}

GPS klasöründeki altı kayıt \texttt{{rfly\_ctrl\_lxl.id=123455}} taşıdığı için
magnetometre arızası olarak karantinaya alınmıştır. Ana fault havuzunda yalnız
\texttt{{ID=123456}} kullanılmıştır. Fit ve calibration yalnız gerçek
\texttt{{Real-No\_Fault}} uçuşlarından yapılmıştır. GPS-fault uçuşlarının
pre-fault bölümleri dahi fit/calibration'a sokulmamıştır.

\section{{Özellikler ve Yöntemler}}
Temel zaman ekseni yaklaşık 2 Hz \texttt{{estimator\_innovations}} mesajıdır.
GPS yatay konum, yatay hız ve dikey hız innovation değerleri ilgili innovation
variance'ın kareköküne bölünmüştür. Test ratio, EKF flag/reset durumları, GPS
fix/EPh/EPv/HDOP/VDOP/satellite/noise/jamming alanları ve local velocity
nedensel backward-asof birleştirme ile eklenmiştir. Forward-fill yapılmamıştır.

\begin{{enumerate}}
  \item \textbf{{PX4-native:}} test ratio $>1$ veya innovation-check flag.
  \item \textbf{{CUSUM:}} beş imzalı normalize innovation kanalında çift yönlü
  Page birikimi; eşik yalnız doğal calibration alarm yükünden seçilmiştir.
  \item \textbf{{Contextual LSTM:}} 12 geçmiş adım, kanal-bazlı location/scale
  tahmini, natural-only robust scaling ve conditional conformal p-değerleri.
\end{{enumerate}}

\section{{Calibration'da Dondurulan Kararlar}}
\begin{{tabular}}{{lrrrr}}
\toprule Sözleşme & CUSUM eşiği & CUSUM alarm/saat & LSTM alpha & LSTM alarm/saat \\
\midrule
{chr(10).join(calibration_rows)}
\bottomrule
\end{{tabular}}

Kritik CUSUM eşiğinin sonsuz seçilmesi bir yazılım hatası değildir: calibration
setinde 2 alarm/saat bütçesini sağlayan sonlu bir eşik bulunamamış ve güvenli
fallback ``alarm üretme'' olmuştur. Aynı eşikler development ve rehearsal
sonuçlarına bakılmadan korunmuştur.

\section{{Development ve Rehearsal Sonuçları}}
\small
\begin{{longtable}}{{lllrrrrrrr}}
\toprule
Rol & Sözleşme & Yöntem & Recall & Wilson alt & Noise & Scale & Alarm/saat & Coverage & Kapı \\
\midrule
\endhead
{chr(10).join(result_rows)}
\bottomrule
\end{{longtable}}
\normalsize

\subsection{{Yöntem Bazında Neden Başarısız Oldu?}}
\begin{{itemize}}
  \item \textbf{{PX4-native:}} rehearsal kritik recall'ı \%90 olsa da doğal
  alarm yükü yaklaşık 28.86/saat olmuştur. Development yükü de yaklaşık
  19.58/saat seviyesindedir. Sinyal vardır fakat operatör yükü kabul edilemez.
  \item \textbf{{CUSUM:}} kritik bütçede eşik alarm üretmeyecek seviyeye çıkmış,
  kritik recall sıfır olmuştur. Advisory'de rehearsal recall \%70 ve
  7.21 alarm/saat olsa da gereken \%90 recall ve fault-mode alt kapıları
  geçilememiştir.
  \item \textbf{{LSTM:}} rehearsal kritik görünümü \%90 recall ve sıfır doğal
  alarmdır; ancak aynı frozen karar development'da yalnız \%47.1 recall ve
  yaklaşık 20.32 alarm/saat üretmiştir. Advisory development yükü yaklaşık
  101.60 alarm/saat olmuştur. Bu, calibration/rehearsal'a özgü başarı ve
  dağılım-kararsızlığıdır; ürün genellemesi değildir.
\end{{itemize}}

\section{{LSTM Büyüklük-Dominasyonu Kontrolü}}
Trained--random Spearman: \textbf{{{_num(diagnostic["trained_vs_random_spearman"], 3)}}};
trained--ham büyüklük Spearman:
\textbf{{{_num(diagnostic["trained_vs_magnitude_spearman"], 3)}}}.
Her ikisi de 0.80 kapısının altındadır; dolayısıyla eski autoencoder turlarındaki
salt büyüklük-dominasyonu bu modelde aynı biçimde tekrarlanmamıştır. Buna rağmen
model alarm-yükü/genelleme kapısını geçememiştir. Başarısızlık nedeni bu turda
``model hiç öğrenmedi'' değil, öğrenilen ilişkinin roller arasında operasyonel
olarak taşınmamasıdır.

\section{{HIL/SIL Rüzgâr Stres Testi}}
Bu 886 vaka GNSS fault ground-truth'u değildir. Yalnız dondurulmuş GNSS
detectorlerinin çevresel bozucuda ürettiği alarm yükünü ölçer.
\begin{{longtable}}{{lllrrr}}
\toprule Domain & Sözleşme & Yöntem & Vaka & Alarm/saat & Coverage \\ \midrule
\endhead
{chr(10).join(stress_rows)}
\bottomrule
\end{{longtable}}

HIL advisory koşulunda CUSUM 20.53 ve LSTM 22.61 alarm/saat üretmiştir.
Bu değerler advisory bütçesi 12/saatin üzerindedir ve çevresel/domain
değişimine hassasiyet göstermektedir. Simülasyon sonucu recall veya ürün
başarısı şeklinde yorumlanmamıştır.

\section{{Motor Pilotunun Neden Yapılmadığı}}
Gerçek RflyMAD loglarında \texttt{{esc\_status}} şeması bulunmasına rağmen RPM,
akım, voltaj ve sıcaklık alanları tüm incelenen normal/motor/sensör
uçuşlarında sabit sıfırdır. PWM actuator output'ları tek başına motor sağlık
truth'u veya gerçek ESC cevabı değildir. Bu nedenle mevcut veriyle motor
sağlığı pilotu yürütmek fiziksel gözlenebilirlik sözleşmesini ihlal ederdi.

\section{{Nihai Karar ve Öneri}}
\begin{{enumerate}}
  \item Kör holdout açılmamalıdır; önceki roller geçilmediği için bilimsel
  açma koşulu oluşmamıştır.
  \item Bu namespace içinde eşik gevşetme, yöntem OR-fusion'ı, yeni model veya
  Optuna araması yapılmamalıdır.
  \item Proje sonucu ``genel detector başarısız'' cümlesiyle sınırlanmamalıdır:
  GNSS innovation sinyali vardır, fakat kabul edilebilir alarm yüküyle
  roller arası taşınabilir bir karar sınırı bulunamamıştır.
  \item Yeni yatırım ancak daha fazla bağımsız normal uçuş, kontrollü ve
  fiziksel olarak belgelenmiş GNSS fault kampanyası ve platforma özgü
  calibration planıyla anlamlıdır.
  \item Motor yönüne dönülecekse önce gerçek ESC RPM/akım/sıcaklık telemetrisi
  ve kontrollü motor fault ground-truth'u sağlanmalıdır.
\end{{enumerate}}

\section{{Tekrar Üretilebilirlik}}
Aktif komut:
\begin{{verbatim}}
python scripts/run_uav_gnss_integrity_v1_locked.py \
  --stage rehearse --config configs/uav_gnss_integrity_v1.json
python scripts/run_uav_gnss_integrity_v1_locked.py \
  --stage stress --config configs/uav_gnss_integrity_v1.json
\end{{verbatim}}
Holdout komutu, frozen config hash'ine bağlı ayrı
\texttt{{HOLDOUT\_UNSEAL.json}} olmadan çalışmaz.

\section{{Kaynak Veri Semantiği}}
RflyMAD resmî veri dokümantasyonu:
\url{{https://rfly-openha.github.io/documents/4_resources/dataset.html}}.
Fault-ID ve parametre sözleşmesi:
\url{{https://rfly-openha.github.io/documents/4_resources/flight_information.html}}.

\end{{document}}
""",
        encoding="utf-8",
    )
    return output

