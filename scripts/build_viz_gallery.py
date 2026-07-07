# -*- coding: utf-8 -*-
"""Gorselleme galerisini uret: artifacts/viz/index.html.

Kendi kendine yeten statik sayfa — tarayicida dogrudan acilir (cift tik),
sunucu gerekmez. Gorseller diskten goreli yolla yuklenir; her figurun
altinda "nasil okunur / ne goruyoruz" aciklamasi vardir. Gorseller
yeniden uretilirse bu script tekrar kosulur.

Kullanim: python scripts/build_viz_gallery.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "artifacts/viz"
OUT = VIZ / "index.html"

DATASETS = {
    "alfa": "ALFA (sabit kanat, mekanik arıza)",
    "uav_attack": "UAV Attack (PX4, siber saldırı)",
    "uav_sead": "UAV-SEAD (PX4)",
}
SECTIONS = {
    "s1_portfolio": "Veri Karnesi",
    "s2_embeddings": "Projeksiyonlar (PCA / t-SNE)",
    "s3_features": "Feature Analitikleri",
    "s4_model": "Model Tanılama",
}

# ---------------------------------------------------------------------------
# Kurasyonlu aciklamalar. Anahtar: "<dataset>/<bolum>/<dosya>".
# Sayilar dogrudan calisma ciktilarindan (H25-H30, viz_manifest'ler).
# ---------------------------------------------------------------------------

E: dict[str, tuple[str, str]] = {}  # key -> (baslik, aciklama)


def _put(dataset: str, rel: str, title: str, text: str) -> None:
    E[f"{dataset}/{rel}"] = (title, text)


# ---- Ortak kaliplar -------------------------------------------------------
for ds, ad in DATASETS.items():
    _put(ds, "s1_portfolio/class_counts.png", f"Sınıf Dağılımı — {ad}",
         "Uçuş düzeyinde etiket sayıları; her barın üstünde n değeri. "
         "Nasıl okunur: barlar arasındaki dengesizlik, hangi sınıflar için genelleme "
         "iddiasının istatistiksel olarak zayıf kalacağını gösterir (küçük-n disiplini). "
         + ("UAV-SEAD'de 131 uçuşluk kör holdout bu sayımın DIŞINDADIR; görülen 480 uçuş "
            "geliştirme kümesidir." if ds == "uav_sead" else
            "Örneklem akademik veri setinin kendi doğal boyutudur; eksik indirme yoktur."))
    _put(ds, "s1_portfolio/completeness_heatmap.png", f"Feature × Etiket Doluluk Haritası — {ad}",
         "Her hücre: o feature kolonunun o etiketteki satırlarda yüzde kaç dolu (non-null) olduğu. "
         "Nasıl okunur: koyu (0'a yakın) satırlar 'bu kategoride bu sinyal fiziksel olarak yok' demektir — "
         "model sorunu değil, veri/kapsam sorunu. "
         + ("UAV-SEAD'de baro tabanlı kolonların düşük doluluğu, Position.Z zafiyetinin kök nedenlerinden "
            "biridir (en iyi ayrıştırıcılar baro-tabanlı ama yalnız ~%7 satırda mevcut)."
            if ds == "uav_sead" else
            "Manuel feature mühendisliğine başlamadan önce ilk bakılacak harita budur."))
    _put(ds, "s1_portfolio/flight_duration_hist.png", f"Uçuş Süresi Dağılımı — {ad}",
         "Uçuş sürelerinin histogramı, uçuş etiketine göre renklendirilmiş (yığılmış). "
         "Nasıl okunur: bir sınıfın uçuşları sistematik olarak daha kısa/uzunsa, uçuş-düzeyi "
         "metrikler süre etkisiyle karışabilir; yanlış-alarm/saat gibi normalize metrikleri "
         "bu yüzden tercih ediyoruz.")
    _put(ds, "s2_embeddings/pca_variance.png", f"PCA Açıklanan Varyans — {ad}",
         "İlk 20 bileşenin tek tek (bar) ve kümülatif (çizgi) açıkladığı varyans. "
         "Nasıl okunur: eğri ne kadar hızlı doyuyorsa veri o kadar düşük boyutlu demektir. "
         "Projeksiyon figürlerinde ölçeklenmiş değerlere ±10 IQR görsel kırpma uygulanır "
         "(tek uç değer PCA'yı tek noktaya sıkıştırıyordu); kırpma hiçbir skor/model hesabına girmez.")
    _put(ds, "s2_embeddings/pca_label.png", f"PCA (2B): Anomali Kategorisi — {ad}",
         "Satır-düzeyi feature vektörlerinin ilk iki ana bileşene izdüşümü; renk = kategori. "
         "Nasıl okunur: PCA global/dogrusal yapıyı korur — büyük ölçekli ayrışmalar gerçektir. "
         "Bir kategori normal bulutunun içine gömülüyse doğrusal olarak ayrıştırılamıyordur.")
    _put(ds, "s2_embeddings/pca_binary.png", f"PCA (2B): Normal vs Anomali — {ad}",
         "Aynı PCA izdüşümü, ikili boyama. Nasıl okunur: anomali noktalarının normal bulutla "
         "örtüşme derecesi, 'tek eşikli' bir dedektörün teorik tavanını sezdirir.")
    _put(ds, "s2_embeddings/pca_fusion.png", f"PCA (2B): IF-Füzyon Skoru — {ad}",
         "Aynı PCA izdüşümü; renk = mevcut IF-füzyon skorunun sürekli değeri (modelin 'uzaklık' algısı). "
         "Nasıl okunur: sıcak renk bölgeleri modelin şüpheli bulduğu bölgeler. Anomali kümelerinin "
         "soğuk kalması = model o bölgeyi normalden ayıramıyor.")
    _put(ds, "s2_embeddings/tsne_label.png", f"t-SNE (2B): Anomali Kategorisi — {ad}",
         "Satır vektörlerinin t-SNE izdüşümü; renk = kategori. ÖNEMLİ: t-SNE yalnız lokal komşuluğu "
         "korur — kümeler ARASI mesafe ve küme BOYUTU yorumlanmaz; yalnız 'hangi noktalar birbirine "
         "komşu' sorusuna bakılır. Ayrı adacık oluşturan kategori ayrıştırılabilir; normal bulutuna "
         "karışan kategori satır-düzeyi feature'larla ayrıştırılamıyordur.")
    _put(ds, "s2_embeddings/tsne_binary.png", f"t-SNE (2B): Normal vs Anomali — {ad}",
         "Aynı t-SNE izdüşümü, ikili boyama. Karışım ne kadar fazlaysa satır-düzeyi tespit o kadar zor; "
         "adalaşma varsa sinyal var demektir. (Kümeler arası mesafe yorumlanmaz.)")
    _put(ds, "s2_embeddings/tsne_fusion.png", f"t-SNE (2B): IF-Füzyon Skoru — {ad}",
         "Aynı t-SNE izdüşümü; renk = kalibre IF-füzyon skoru. Nasıl okunur: anomali adacıkları "
         "sıcak renkteyse model onları görüyordur; soğuksa 'feature var ama model kullanamıyor' "
         "vakasıdır (feature analitikleriyle birlikte okuyun).")
    _put(ds, "s3_features/feature_auc_heatmap.png", f"Feature–Kategori Ayrıştırma (AUC) — {ad}",
         "ANA ÇIKTI. Her hücre: o feature'ın o kategoriyi normal satırlardan TEK BAŞINA ne kadar "
         "ayırdığı (Mann-Whitney AUC; 0.5 = ayrışma yok, 1'e/0'a yaklaştıkça güçlü). Nasıl okunur: "
         "(1) bir kategorinin sütunu baştan aşağı soluksa → hiçbir feature ayırmıyor → VERİ/kapsam "
         "sorunu; (2) güçlü hücre var ama füzyon o kategoride zayıfsa → MODEL/füzyon sorunu (seyrelme) "
         "→ gerçek feature-engineering adayı. Bu ayrım, iyileştirme çabasının nereye harcanacağını "
         "belirler. Dikkat: bazı yüksek AUC'ler arıza fiziği değil uçuş-profili karışıklığı olabilir "
         "(ör. arızalı uçuşun eve dönmesi).")
    _put(ds, "s3_features/feature_q99_heatmap.png", f"Seyrek İmza Göstergesi (q99 oranı) — {ad}",
         "Her hücre: log2(kategori q99 / normal q99), |değer| üzerinden. Nasıl okunur: ortalama-tabanlı "
         "metriklerin kaçırdığı 'seyrek uç değer' imzalarını yakalar — AUC ılımlı ama q99 oranı "
         "yüksekse sinyal nadir-ama-şiddetli demektir (CUSUM/max tipi dedektörlere aday). CUSUM-tipi "
         "feature'larda normal q99≈0 olduğu için oran patlayabilir; tek başına değil AUC ile birlikte okuyun.")
    _put(ds, "s3_features/spearman_heatmap.png", f"Feature Korelasyonu (Spearman) — {ad}",
         "Feature×feature sıra korelasyonu; eksenler hiyerarşik kümelemeyle sıralı (bloklar = birbirinin "
         "kopyası feature aileleri). Nasıl okunur: koyu kırmızı/mavi bloklar bilgi tekrarıdır — modele "
         "hepsini vermek sinyali sulandırabilir (ML-12'de bunun kategori recall'una maliyeti ölçüldü). "
         "|ρ|>0.9 çiftleri redundant_pairs.csv dosyasındadır.")
    _put(ds, "s3_features/feature_auc_matrix.csv", f"feature_auc_matrix.csv — {ad}",
         "AUC haritasının ham tablosu: feature, kategori, auc, separation, q99_abs_ratio, n_pos/n_neg. "
         "Top-10 manuel feature-engineering aday listesi bu tablodan türetildi "
         "(docs/ML1_BULGULAR_VE_HATALAR.md, 'Görselleştirme sonuçları').")
    _put(ds, "s3_features/redundant_pairs.csv", f"redundant_pairs.csv — {ad}",
         "|ρ|>0.9 olan feature çiftleri (Spearman). Modül tasarımında aynı çiftten ikisini birden "
         "kullanmak bilgi eklemez, seyreltir.")
    _put(ds, "s1_portfolio/completeness_matrix.csv", f"completeness_matrix.csv — {ad}",
         "Doluluk haritasının ham tablosu (feature × etiket non-null oranı).")

# ---- Dataset'e ozel -------------------------------------------------------
_put("uav_sead", "s1_portfolio/session_histogram.png",
     "Normal Uçuşların Oturum Dağılımı — UAV-SEAD",
     "Heterojen-normal bulgusunun görsel kanıtı: geliştirme kümesindeki 324 normal uçuş yalnız 49 "
     "oturuma dağılıyor (en büyük oturum 21 uçuş); tüm sette 398 uçuş / 64 oturum. Nasıl okunur: "
     "gerçek bağımsız örneklem uçuş sayısı değil oturum sayısına yakındır — 'normali öğren' hedefi "
     "tek bir yoğunluk değil, oturum ailelerinin birleşimidir. Bu, veri artsa bile normal sınıfın "
     "kolaylaşmamasının nedenidir. (Not: önceki kayıtlardaki '~32 oturum' tahmini bu ölçümle 64 "
     "olarak düzeltildi.)")
_put("uav_sead", "s1_portfolio/annotation_event_counts.png",
     "Anotasyon Kategorisi Başına Aralık Sayısı — UAV-SEAD",
     "Geliştirme kümesinde her anomali kategorisi için etiketli zaman aralığı (event) sayısı. "
     "Nasıl okunur: event sayısı, kategori-recall metriklerinin paydasıdır; az-eventli kategorilerde "
     "(Battery, Actuator Thrust, Velocity) recall tek event'le %20-30 oynayabilir — o sütunlar "
     "yorumlanırken n mutlaka dikkate alınır.")
_put("uav_sead", "s2_embeddings/pca_session.png",
     "PCA (2B): Normal Uçuşların Oturumu — UAV-SEAD",
     "Yalnız normal uçuş noktaları, renk = oturum (49 oturum); anomalili uçuşlar soluk gri. "
     "Nasıl okunur: renklerin ayrı bölgelerde toplanması, normal sınıfın oturum bazında "
     "parçalandığını gösterir.")
_put("uav_sead", "s2_embeddings/tsne_session.png",
     "t-SNE (2B): Normal Uçuşların Oturumu — UAV-SEAD",
     "Heterojen-normal tezinin en net görseli: normal noktalar tek bir bulut değil, büyük ölçüde "
     "OTURUM rengini izleyen adacıklar. Nasıl okunur: yeni bir oturumun uçuşu, eğitimde görülen "
     "adaların hiçbirine düşmeyebilir — normal-sınıfı modellerinin (IF/LSTM-AE) yanlış alarm "
     "kaynağı budur. Oturum-koşullu model bilinçli olarak REDDEDİLDİ (yeni oturumda referans "
     "kaybı); bu görsel sınırın kendisini belgeler, çözüm önerisi değildir.")
_put("uav_sead", "s4_model/score_violin.png",
     "Skor Dağılımı: Normal vs Kategoriler — UAV-SEAD",
     "Satır düzeyi kalibre IF-füzyon skoru (split_00), kategori bazında violin. Nasıl okunur: "
     "füzyon 6 modülün maksimumu olduğu için taban yukarı itilir — normal satırlar bile 0.92-1.0 "
     "bandındadır (doygunluk). Kategori violinleri normalle büyük ölçüde örtüşüyor: kategori-recall "
     "kaybının bir kısmı skor değil KARAR MARJI sorunudur. Battery/Velocity'nin ayrışması küçük "
     "n ile birlikte okunmalıdır.")
_put("uav_sead", "s4_model/roc_pr_flight.png",
     "Uçuş Düzeyi ROC / PR — UAV-SEAD",
     "IF-füzyonun uçuş-düzeyi ROC ve Precision-Recall eğrileri; 5 seed ortalaması, bant = min-max. "
     "Nasıl okunur: ROC AUC ≈ 0.557 — SEAD'de ayrım uçuş düzeyinde DEĞİL event düzeyinde yaşanır "
     "(uçuşun tamamı değil, içindeki kısa anomali aralıkları etiketlidir); sistemin asıl metrikleri "
     "event-onset recall ve FA/saat'tir. Bu figür 'uçuş sınıflandırıcı' beklentisini kalibre etmek "
     "için vardır.")
_put("uav_sead", "s4_model/sead_confusion_binary.png",
     "Uçuş Düzeyi Karar Matrisi (2×2) — UAV-SEAD",
     "Advisory CUSUM çalışma noktasında, 5 seed toplamı: gerçek normal/anomali × alarm var/yok. "
     "NOT: model sınıf tahmin etmez; bu ikili ALARM kararıdır. Nasıl okunur: TP 416 / FN 364 / "
     "FP 380 / TN 543 — normal uçuşların yaklaşık %41'inde en az bir yanlış alarm var; advisory "
     "FA bütçesinin neden aşıldığının uçuş-düzeyi görünümü.")
_put("uav_sead", "s4_model/sead_confusion_by_type.png",
     "Tür Bazlı Tespit Matrisi — UAV-SEAD",
     "Unsupervised ikili dedektörde çok-sınıflı confusion matrix'in doğru karşılığı: satır = gerçek "
     "anomali türü, sütun = tespit/kaçırma. Nasıl okunur: external_position 199/225 ve "
     "global_position 99/110 güçlü; altitude 60/295 ve mechanical 58/150 zayıf — feature "
     "analitikleri haritasıyla birebir tutarlı (altitude → veri/kapsam sorunu; mechanical → "
     "seyrelme, ML-12'de ince modülle 0.459'a çıkarıldı). 'normal' satırı yanlış alarm görünümüdür.")
_put("alfa", "s4_model/score_violin.png",
     "Pencere Skoru Dağılımı — ALFA",
     "LSTM-AE pencere skorları (log10), etiket bazında; kesikli çizgi = val-q99 pencere eşiği. "
     "Nasıl okunur: anomali violinlerinin gövdesi eşiğin altında kalıyorsa model sıralamayı bilse "
     "de ÇALIŞMA NOKTASI muhafazakâr demektir — ALFA'da eşik yalnız 2 normal val uçuşundan "
     "kalibre edilmek zorunda (küçük-n eşik kararsızlığı, H2).")
_put("alfa", "s4_model/roc_pr_flight.png",
     "Uçuş Düzeyi ROC / PR — ALFA",
     "LSTM-AE (AUC 0.750) vs IF-füzyon (0.622); tek paketlenmiş model olduğu için bant yok. "
     "Nasıl okunur: sıralama gücü var (0.75) ama aşağıdaki karar matrisiyle birlikte okununca "
     "eşiğin bu gücü operasyona çeviremediği görülür.")
_put("alfa", "s4_model/alfa_confusion_binary.png",
     "Uçuş Düzeyi Karar Matrisi (2×2) — ALFA",
     "LSTM-AE, val-q99 pencere eşiği çalışma noktası. Nasıl okunur: 38 anomalili uçuştan yalnız 3'ü "
     "alarm üretti (0 yanlış alarmla) — eşik aşırı muhafazakâr. ROC 0.750 ile bu tablo arasındaki "
     "fark, 'model mi zayıf, eşik mi' sorusunun cevabıdır: eşik. Kök neden: eşik yalnız 2 normal "
     "val uçuşundan kalibre edilebiliyor.")
_put("alfa", "s4_model/alfa_confusion_by_type.png",
     "Tür Bazlı Tespit Matrisi — ALFA",
     "Satır = arıza türü, sütun = tespit/kaçırma (LSTM-AE, val-q99 eşiği). Nasıl okunur: n değerleri "
     "küçüktür (rudder 3, elevator 2, aileron_rudder 1 uçuş) — tek uçuş %33-100 oynatır; bu matris "
     "eğilim gösterir, genelleme iddiası taşımaz.")

# ---- UAV Attack ozel notlari ----------------------------------------------
_put("uav_attack", "s2_embeddings/tsne_label.png",
     "t-SNE (2B): Anomali Kategorisi — UAV Attack",
     "19 uçuşluk küçük set (6 benign / 6 spoofing / 6 ping_dos / 1 jamming). Nasıl okunur: "
     "spoofing satırları genelde ayrışır (GPS tutarlılık residual'ları), ping_dos'un büyük kısmı "
     "normale karışır — ağ-katmanı imzası bu telemetri topic'lerine yansımıyor (yapısal sınır A.5). "
     "Kümeler arası mesafe yorumlanmaz.")

# ---- Zaman serisi kaliplari ------------------------------------------------
TS_SEAD = ("Zaman Serisi: {cat} — {flight}",
           "Tek uçuşun 1 sn'lik IF-füzyon skor akışı (split_00). Kırmızı bant = etiketli gerçek "
           "anomali aralığı ('nerede çalmalıydı'); kırmızı üçgen = advisory CUSUM alarm başlangıcı "
           "('nerede çaldı'). Nasıl okunur: bant içinde üçgen yoksa kaçırma; bant dışındaki üçgenler "
           "yanlış alarmdır. Skorun 0.92-1.0 bandında sıkışması füzyon doygunluğunun (H27) uçuş "
           "içindeki görünümüdür.")
TS_ALFA = ("Zaman Serisi: {cat} — {flight}",
           "Tek uçuşun LSTM-AE pencere skoru (log10). Kesikli çizgi = val-q99 eşiği; kırmızı bant = "
           "arıza bölgesi (etiket onset'ten uçuş sonuna). Nasıl okunur: skor bant içinde yükselip "
           "eşiği aşmıyorsa model arızayı 'hissediyor ama alarm veremiyor' demektir — eşik "
           "muhafazakârlığının (H28) uçuş içindeki görünümü.")


def _timeseries_entry(dataset: str, rel: str) -> tuple[str, str]:
    name = Path(rel).stem  # timeseries_<cat>_<flight>
    body = name[len("timeseries_"):]
    if dataset == "uav_sead":
        match = re.match(r"(.+?)_(\d{4}-\d{2}-\d{2}__.+|log_.+)$", body)
        cat = (match.group(1) if match else body).replace("_", " ")
        flight = (match.group(2) if match else "").replace("__", "/")
        tpl = TS_SEAD
    else:
        parts = body.split("_carbonZ_", 1)
        cat = parts[0]
        flight = "carbonZ_" + parts[1] if len(parts) == 2 else body
        tpl = TS_ALFA
    return tpl[0].format(cat=cat, flight=flight), tpl[1]


def collect_items() -> list[dict]:
    items = []
    missing = []
    for dataset in DATASETS:
        base = VIZ / dataset
        for path in sorted(base.rglob("*")):
            if path.suffix not in {".png", ".csv"} or path.name == "index.html":
                continue
            rel = str(path.relative_to(base)).replace("\\", "/")
            section = rel.split("/", 1)[0]
            key = f"{dataset}/{rel}"
            if key in E:
                title, text = E[key]
            elif Path(rel).name.startswith("timeseries_"):
                title, text = _timeseries_entry(dataset, rel)
            else:
                missing.append(key)
                title, text = Path(rel).stem, ""
            items.append({
                "dataset": dataset,
                "section": section,
                "path": f"{dataset}/{rel}",
                "kind": "csv" if path.suffix == ".csv" else "png",
                "title": title,
                "text": text,
            })
    if missing:
        raise SystemExit("Aciklamasi olmayan dosyalar var:\n" + "\n".join(missing))
    return items


PAGE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Görselleştirme Galerisi — Anomali Tespiti</title>
<style>
:root {
  --bg: #f6f7f9; --card: #ffffff; --ink: #1c2733; --muted: #5b6b7a;
  --line: #dde3ea; --accent: #1f6feb; --chip: #eef2f6; --chip-on: #1f6feb;
  --chip-on-ink: #ffffff; --shadow: 0 1px 3px rgba(16,24,32,.08);
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#12161b; --card:#1a2027; --ink:#e6edf3; --muted:#9db0c0;
          --line:#2b3540; --accent:#539bf5; --chip:#232c35; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.55 "Segoe UI", system-ui, sans-serif; }
header { padding:26px 28px 14px; border-bottom:1px solid var(--line); }
header h1 { margin:0 0 6px; font-size:22px; font-weight:650; }
header p  { margin:0; color:var(--muted); max-width:70ch; }
.note { margin:10px 28px 0; padding:9px 14px; background:var(--chip);
        border:1px solid var(--line); border-radius:8px; color:var(--muted);
        font-size:13.5px; max-width:fit-content; }
.controls { position:sticky; top:0; z-index:5; background:var(--bg);
            padding:12px 28px; border-bottom:1px solid var(--line);
            display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.chip { border:1px solid var(--line); background:var(--chip); color:var(--ink);
        padding:5px 13px; border-radius:999px; cursor:pointer; font-size:13.5px; }
.chip.on { background:var(--chip-on); color:var(--chip-on-ink);
           border-color:var(--chip-on); }
.controls .sep { width:1px; height:22px; background:var(--line); margin:0 4px; }
#q { margin-left:auto; padding:6px 12px; border:1px solid var(--line);
     border-radius:8px; background:var(--card); color:var(--ink);
     min-width:230px; font-size:14px; }
#count { color:var(--muted); font-size:13px; }
main { padding:20px 28px 60px; display:grid; gap:18px;
       grid-template-columns:repeat(auto-fill, minmax(430px, 1fr)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
        overflow:hidden; box-shadow:var(--shadow); display:flex;
        flex-direction:column; }
.card .imgwrap { background:#fff; border-bottom:1px solid var(--line);
                 cursor:zoom-in; display:flex; justify-content:center; }
.card img { max-width:100%; max-height:330px; object-fit:contain; display:block; }
.card .body { padding:13px 16px 15px; }
.card h3 { margin:0 0 6px; font-size:15px; font-weight:640; }
.card p  { margin:0; color:var(--muted); font-size:13.5px; }
.tags { display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; }
.tag { font-size:11.5px; padding:2px 9px; border-radius:999px;
       background:var(--chip); color:var(--muted); border:1px solid var(--line); }
.csvcard .imgwrap { padding:26px; font-size:34px; cursor:default; }
.csvcard a { color:var(--accent); text-decoration:none; font-weight:600; }
#lb { position:fixed; inset:0; background:rgba(10,14,18,.93); display:none;
      z-index:50; flex-direction:column; }
#lb.open { display:flex; }
#lb .stage { flex:1; display:flex; align-items:center; justify-content:center;
             min-height:0; padding:14px; }
#lb img { max-width:100%; max-height:100%; object-fit:contain; background:#fff;
          border-radius:6px; }
#lb .cap { color:#dfe8f0; padding:10px 60px 22px; text-align:center;
           max-width:110ch; margin:0 auto; font-size:14px; }
#lb .cap b { display:block; font-size:16px; margin-bottom:4px; }
#lb button { position:absolute; background:none; border:none; color:#dfe8f0;
             font-size:34px; cursor:pointer; padding:12px 18px; }
#lb .x { top:6px; right:10px; } #lb .prev { left:4px; top:45%; }
#lb .next { right:4px; top:45%; }
</style>
</head>
<body>
<header>
  <h1>Görselleştirme Galerisi — Veri Keşfi ve Model Tanılama</h1>
  <p>Amaç: "bazı arıza türleri neden hâlâ zayıf tespit ediliyor?" sorusuna görsel kanıt.
     Karta tıklayınca büyür; ←/→ ile gezilir, Esc ile kapanır.</p>
</header>
<div class="note">Bütünlük notu: UAV-SEAD'in 131 uçuşluk kör holdout'u hiçbir görselde/istatistikte yoktur;
tüm sayılar geliştirme kümesindendir ve deterministiktir (seed=42). Kaynak: <code>scripts/make_visualizations.py</code>,
checksum: <code>viz_manifest.json</code>.</div>
<div class="controls">
  <span id="dsChips"></span><span class="sep"></span>
  <span id="secChips"></span>
  <input id="q" type="search" placeholder="Ara: başlık/açıklama…">
  <span id="count"></span>
</div>
<main id="grid"></main>
<div id="lb"><button class="x" title="Kapat">×</button>
  <button class="prev" title="Önceki">‹</button>
  <div class="stage"><img alt=""></div>
  <button class="next" title="Sonraki">›</button>
  <div class="cap"></div>
</div>
<script>
const ITEMS = __ITEMS__;
const DS = __DS__;
const SEC = __SEC__;
let ds = "all", sec = "all", q = "";
const grid = document.getElementById("grid");

function chipbar(el, defs, get, set) {
  const all = {all: "Tümü", ...defs};
  el.innerHTML = "";
  for (const [key, label] of Object.entries(all)) {
    const b = document.createElement("button");
    b.className = "chip" + (get() === key ? " on" : "");
    b.textContent = label;
    b.onclick = () => { set(key); render(); };
    el.appendChild(b);
  }
}
function visible() {
  const needle = q.toLocaleLowerCase("tr");
  return ITEMS.filter(it =>
    (ds === "all" || it.dataset === ds) &&
    (sec === "all" || it.section === sec) &&
    (!needle || (it.title + " " + it.text).toLocaleLowerCase("tr").includes(needle)));
}
function render() {
  chipbar(document.getElementById("dsChips"), DS, () => ds, v => ds = v);
  chipbar(document.getElementById("secChips"), SEC, () => sec, v => sec = v);
  const rows = visible();
  document.getElementById("count").textContent = rows.length + " öğe";
  grid.innerHTML = "";
  rows.forEach((it, i) => {
    const card = document.createElement("div");
    card.className = "card" + (it.kind === "csv" ? " csvcard" : "");
    const tags = `<div class="tags"><span class="tag">${DS[it.dataset]}</span>` +
                 `<span class="tag">${SEC[it.section]}</span></div>`;
    if (it.kind === "png") {
      card.innerHTML = `<div class="imgwrap"><img loading="lazy" src="${it.path}" alt=""></div>` +
        `<div class="body"><h3>${it.title}</h3><p>${it.text}</p>${tags}</div>`;
      card.querySelector(".imgwrap").onclick = () => openLb(rows.filter(r => r.kind === "png"),
        rows.filter(r => r.kind === "png").indexOf(it));
    } else {
      card.innerHTML = `<div class="imgwrap">🗒️</div>` +
        `<div class="body"><h3><a href="${it.path}" download>${it.title}</a></h3>` +
        `<p>${it.text}</p>${tags}</div>`;
    }
    grid.appendChild(card);
  });
}
const lb = document.getElementById("lb");
let lbRows = [], lbIdx = 0;
function openLb(rows, idx) { lbRows = rows; lbIdx = idx; showLb(); lb.classList.add("open"); }
function showLb() {
  const it = lbRows[lbIdx];
  lb.querySelector("img").src = it.path;
  lb.querySelector(".cap").innerHTML = `<b>${it.title}</b>${it.text}`;
}
function move(d) { lbIdx = (lbIdx + d + lbRows.length) % lbRows.length; showLb(); }
lb.querySelector(".x").onclick = () => lb.classList.remove("open");
lb.querySelector(".prev").onclick = () => move(-1);
lb.querySelector(".next").onclick = () => move(1);
lb.onclick = e => { if (e.target === lb || e.target.classList.contains("stage")) lb.classList.remove("open"); };
document.addEventListener("keydown", e => {
  if (!lb.classList.contains("open")) return;
  if (e.key === "Escape") lb.classList.remove("open");
  if (e.key === "ArrowLeft") move(-1);
  if (e.key === "ArrowRight") move(1);
});
document.getElementById("q").addEventListener("input", e => { q = e.target.value; render(); });
render();
</script>
</body>
</html>
"""


def main() -> None:
    items = collect_items()
    page = (PAGE
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
            .replace("__DS__", json.dumps(DATASETS, ensure_ascii=False))
            .replace("__SEC__", json.dumps(SECTIONS, ensure_ascii=False)))
    OUT.write_text(page, encoding="utf-8")
    print(f"galeri yazildi: {OUT} ({len(items)} oge)")


if __name__ == "__main__":
    main()
