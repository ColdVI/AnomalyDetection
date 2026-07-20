"""app.py'den cikarildi, adim 1 -- renk/stil sabitleri, saf veri, yan etki yok."""

# Harita katmani secenekleri -- ayarlardan degistirilebilir. "dark"
# varsayilan (mevcut OpenStreetMap katmani, davranis degismiyor).
#
# GECMIS: Once Esri World Imagery denendi (server.arcgisonline.com) --
# kullanicinin agindan gri ekran cikti, AYNI sorun adsb.lol'un KENDI Esri
# katmaninda da gorulduyu icin bu bizim kodumuzdaki bir hata degildi
# (URL, leaflet-providers'daki kanonik Esri adresiyle birebir ayniydi).
# Google Satellite'e gecildi ama O DA gri cikti -- bu sefer GERCEK bir kod
# hatasiydi: iki katman arasinda "subdomains" degerini de (street icin
# a/b/c, google icin mt0-mt3) dinamik degistirmeye calisiyorduk, ama
# react-leaflet/dash-leaflet TileLayer'da sadece "url" prop'u calisma
# zamaninda guvenilir sekilde uygulaniyor (Leaflet'in setUrl() metoduyla);
# "subdomains" ise SADECE ILK YUKLEMEDE okunuyor, sonradan degisse de
# Leaflet tarafinda yeniden uygulanmiyor. Sonuc: "Uydu"ya gecilince URL
# degisiyordu ama subdomain hala ilk yuklemedeki 'a/b/c' kaliyordu --
# "a.google.com/vt/..." gibi GECERSIZ adreslere istek atiliyordu (Google'in
# gercek subdomain'leri mt0-mt3), hepsi basarisiz oluyordu -> gri ekran.
#
# COZUM: "{s}" sablonunu tamamen kaldirip SABIT tek bir subdomain
# kullanmaya gecildi -- boylece dinamik "subdomains" prop'una hic ihtiyac
# kalmiyor, bu hata sinifi kokten ortadan kalkiyor (paralel-istek
# optimizasyonu kaybediliyor ama bizim trafik hacmimizde onemsiz).
TILE_LAYERS = {
    "street": {
        "url": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap",
    },
    "satellite": {
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attribution": "Map data © Google",
    },
    # CARTO Dark Matter -- API anahtari gerektirmiyor, "satellite"teki AYNI
    # sebeple (bkz. yukaridaki {s} subdomain yorumu) TEK sabit subdomain
    # ("a.basemaps...") kullaniliyor, dinamik subdomain rotasyonu YOK.
    "dark": {
        "url": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors © CARTO",
    },
}
DEFAULT_MAP_STYLE = "dark"

LEFT_PANEL_BASE = {
    "position": "absolute", "top": 0, "left": 0, "bottom": 0,
    "width": "320px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "boxShadow": "4px 0 24px rgba(0,0,0,0.6)",
    "transition": "transform 0.3s ease",
    "zIndex": 800, "padding": "18px", "overflowY": "auto",
}
# "Onceki Ucuslar" listesindeki her satir icin -- secili olan (haritada
# su an gosterilen segment) FILTER_BTN'lerle AYNI aktif/pasif desenini
# kullanir (bkz. style_flight_segment_buttons).
FLIGHT_SEGMENT_BTN_STYLE = {
    "width": "100%", "textAlign": "left", "padding": "6px 8px",
    "borderRadius": "5px", "border": "1px solid #2a2a4a",
    "backgroundColor": "#161625", "cursor": "pointer", "marginBottom": "4px",
}
FLIGHT_SEGMENT_BTN_ACTIVE_STYLE = {**FLIGHT_SEGMENT_BTN_STYLE,
    "border": "1px solid #00b4d8", "backgroundColor": "#0d2830"}
HISTORY_PANEL_BASE = {
    "position": "absolute", "bottom": 0, "right": 0,
    "width": "680px", "height": "300px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "boxShadow": "-2px -2px 24px rgba(0,0,0,0.6)",
    "borderTopLeftRadius": "10px",
    "transition": "transform 0.3s ease",
    "zIndex": 800, "padding": "12px",
}
SETTINGS_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "12px",
    "width": "230px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #2a2a4a",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
STATS_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "60px",
    "width": "420px", "height": "260px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #2a2a4a",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
EMERGENCY_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "108px",
    "width": "300px", "maxHeight": "320px", "overflowY": "auto",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #e63946",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
EMERGENCY_ROW_STYLE = {
    "width": "100%", "textAlign": "left", "padding": "8px 10px",
    "borderRadius": "5px", "border": "1px solid #e63946",
    "backgroundColor": "#2a0f13", "cursor": "pointer", "marginBottom": "6px",
    "color": "#fff",
}
REPLAY_PANEL_BASE = {
    "position": "absolute", "top": "60px", "right": "156px",
    "width": "300px",
    "backgroundColor": "rgba(15,15,25,0.97)",
    "border": "1px solid #2a2a4a",
    "borderRadius": "10px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.5)",
    "padding": "14px", "zIndex": 900,
}
REPLAY_COLOR = "#f7b731"  # canli haritadaki (#00b4d8) mavi/askeri yesilden AYRI -- kullanici
                          # bunun bir "gecmis" gorunum oldugunu tek bakista ayirt etsin

# Gecmis grafigi "Hesapla" butonu -- tarih araligi dropdown'lari artik
# Input DEGIL State (bkz. update_history) -- secim yapmak TEK BASINA
# grafigi guncellemiyor, kullanici bu butona basana kadar bekliyor.
# Boylece 4 dropdown'u tek tek secerken (gun/saat x baslangic/bitis) her
# ara adimda gereksiz sorgu atilmiyor, sadece kullanici hazir oldugunda.
HISTORY_CALC_BTN_STYLE = {
    "padding": "6px 12px", "borderRadius": "5px", "border": "1px solid #00b4d8",
    "backgroundColor": "#00b4d8", "color": "#07070e", "fontSize": "11px",
    "fontWeight": "700", "cursor": "pointer", "flexShrink": 0, "whiteSpace": "nowrap",
}
HISTORY_DOWNLOAD_BTN_STYLE = {
    "padding": "6px 10px", "borderRadius": "5px", "border": "1px solid #2a2a4a",
    "backgroundColor": "#161625", "color": "#c8d0e0", "fontSize": "13px",
    "cursor": "pointer", "flexShrink": 0,
}

# Dil secim butonlari (TR/EN) -- iki durumlu (aktif/pasif) stil, hangisinin
# secili oldugu update_language_buttons callback'inde belirleniyor.
LANG_BTN_BASE_STYLE = {
    "flex": "1", "padding": "6px 0", "borderRadius": "5px",
    "border": "1px solid #2a2a4a", "fontSize": "12px", "fontWeight": "600",
    "cursor": "pointer", "letterSpacing": "0.5px",
}
LANG_BTN_ACTIVE_STYLE = {**LANG_BTN_BASE_STYLE,
    "backgroundColor": "#00b4d8", "color": "#07070e", "border": "1px solid #00b4d8"}
LANG_BTN_INACTIVE_STYLE = {**LANG_BTN_BASE_STYLE,
    "backgroundColor": "#161625", "color": "#888"}

# Askeri ucaklari haritada ayirt etmek icin ayri bir renk -- alarm kirmizisi
# (#e63946) ve varsayilan sivil rengiyle (#00b4d8) karismasin diye hakiki/
# zeytin yesili secildi. Oncelik sirasi: alarm > askeri > sivil (bkz.
# update_map, bir ucak hem alarmli hem askeri olabilir, alarm once gelir).
DEFAULT_AIRCRAFT_COLOR = "#00b4d8"
MILITARY_COLOR = "#8a9a5b"
ALERT_COLOR = "#e63946"

# ONEMLI (kullanici geri bildirimi -- bkz. proje sohbet gecmisi: "900ler
# hala 2000den 4000e geçerken gidiyor"): duraklar ESKIDEN feet cinsindeydi
# (adsb.lol/tar1090 esintili), ama ucak bilgi paneli irtifayi HER YERDE
# METRE gosteriyor -- iki farkli birimin ayni sayilarmis gibi (900 ile
# 2000/4000) karsilastirilmasi kafa karistiriyordu (900m ~= 2953ft, yani
# GERCEKTEN 2000ft-4000ft arasinda kaliyordu -- filtre matematigi dogruydu,
# birim gosterimi tutarsizdi). COZUM: TUM sistem (duraklar, kaydirici,
# lejant, renk esikleri) artik METRE uzerinden calisiyor, ft<->m cevirisi
# TAMAMEN kaldirildi -- gosterilen sayi ile ic hesaplama HER ZAMAN ayni
# birimde. Duraklarin SIKLIGI yine dusuk irtifada YUKSEK, yuksek irtifada
# DUSUK (alcak irtifadaki -- havaalani yakini, trafik yogun -- renk
# degisimini daha hassas gostermek icin).
ALTITUDE_COLOR_STOPS = [
    (0,     "#e8551f"),
    (200,   "#ed7a1f"),
    (500,   "#f0971f"),
    (1000,  "#e8c81f"),
    (2000,  "#c8d820"),
    (3000,  "#78c840"),
    (4000,  "#38b868"),
    (6000,  "#20a8a0"),
    (8000,  "#2078c8"),
    (10000, "#3050d8"),
    (12000, "#9040c8"),  # 12000m+ -- legend'de "12 000+" olarak sabit
]

# ONEMLI: yerde/havada askeri-sivil ekseninden BAGIMSIZ, ayri bir boyut --
# bir ucak ayni anda hem askeri hem yerde olabilir. Bu yuzden GROUND_COLOR
# rengi DEGISTIRMEZ (oncelik hala alarm > askeri > sivil), sadece tooltip'e
# "Yerde" etiketi ekler (bkz. update_map) -- ayri bir renk yerine filtre
# butonunun kendisi (asagida) tarafsiz bir kum/toprak tonu kullanir.
GROUND_COLOR = "#e0a458"

# Sol-ust askeri/sivil filtre butonlari -- haritanin kendi zoom (+/-)
# kontrolu de sol-ustte oldugu icin (Leaflet varsayilani, ~10px kenar
# bosluklu, iki dugme ~52px yukseklik), bu butonlar bilerek onun ALTINA
# (top: 72px) yerlestiriliyor, ustune degil -- cakismalari onlemek icin.
FILTER_BTN_BASE_STYLE = {
    "width": "92px", "padding": "6px 8px", "borderRadius": "6px",
    "fontSize": "11px", "fontWeight": "600", "cursor": "pointer",
    "textAlign": "center", "letterSpacing": "0.3px",
}
FILTER_BTN_CIVIL_ACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": DEFAULT_AIRCRAFT_COLOR, "color": "#07070e",
    "border": f"1px solid {DEFAULT_AIRCRAFT_COLOR}"}
FILTER_BTN_MILITARY_ACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": MILITARY_COLOR, "color": "#07070e",
    "border": f"1px solid {MILITARY_COLOR}"}
FILTER_BTN_GROUND_ACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": GROUND_COLOR, "color": "#07070e",
    "border": f"1px solid {GROUND_COLOR}"}
FILTER_BTN_INACTIVE_STYLE = {**FILTER_BTN_BASE_STYLE,
    "backgroundColor": "#161625", "color": "#888", "border": "1px solid #2a2a4a"}
