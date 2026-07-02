// app.py'deki CATEGORY_LABELS / EMERGENCY_LABELS ile birebir ayni --
// tek kaynak backend'te, burada sadece UI etiketleri icin kopyalandi.
export const CATEGORY_LABELS: Record<string, string> = {
  A0: "Bilinmiyor",
  A1: "Hafif uçak",
  A2: "Küçük uçak",
  A3: "Büyük uçak",
  A4: "Büyük uçak (yüksek vorteks)",
  A5: "Ağır uçak",
  A6: "Yüksek performans",
  A7: "Helikopter",
  B0: "Bilinmiyor",
  B1: "Planör",
  B2: "Balon/Zeplin",
  B3: "Paraşütçü",
  B4: "Ultralight/Yamaç paraşütü",
  B6: "İHA/Drone",
  B7: "Uzay aracı",
  C0: "Bilinmiyor",
  C1: "Yer taşıtı (acil)",
  C2: "Yer taşıtı (servis)",
  C3: "Sabit engel",
  C4: "Engel kümesi",
  C5: "Hat engeli",
};

export const EMERGENCY_LABELS: Record<string, string | null> = {
  none: null,
  general: "GENEL ACİL DURUM",
  lifeguard: "SAĞLIK ACİL DURUMU",
  minfuel: "YAKIT KRİTİK",
  nordo: "RADYO ARIZASI",
  unlawful: "KAÇIRMA (HİJACK)",
  downed: "DÜŞTÜ/İNİŞ ZORUNLU",
  reserved: "REZERVE KOD",
};

export const POLL_INTERVAL_MS = 15000; // Yusuf'un adsb_producer.py'siyle ayni cadence
