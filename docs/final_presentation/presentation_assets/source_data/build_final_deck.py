"""Build the final 35-slide .pptx from the audited slide script.

White theme, single navy accent, one figure per slide, speaker notes carried
into the notes pane. Figures come from
`docs/final_presentation/presentation_assets/final_deck/` (rendered by
render_final_deck.py) plus two real research artifacts from `existing/`.

This script only lays out content. It trains nothing and computes no metric.
Numbers live in the SLIDES table below and every one of them traces to an
audited source listed in docs/final_presentation/12_v3_denetim_ve_yeniden_yazim.md.

Output: docs/final_presentation/Final_Sunum_IHA_Anomali_Tespiti_v4.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[4]
FIG = ROOT / "docs/final_presentation/presentation_assets/final_deck"
EXISTING = ROOT / "docs/final_presentation/presentation_assets/existing"
OUT = ROOT / "docs/final_presentation/Final_Sunum_IHA_Anomali_Tespiti_v4.pptx"

# Same palette as render_final_deck.py so slide chrome and figures agree.
NAVY = RGBColor(0x1C, 0x5C, 0xAB)    # blue-550, headings
BLUE = RGBColor(0x2A, 0x78, 0xD6)    # slot 1
ORANGE = RGBColor(0xEB, 0x68, 0x34)  # slot 2
AQUA = RGBColor(0x1B, 0xAF, 0x7A)    # slot 3
VIOLET = RGBColor(0x4A, 0x3A, 0xA7)
RED = RGBColor(0xD0, 0x3B, 0x3B)
INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
GREY = RGBColor(0x89, 0x87, 0x81)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
TINT_BLUE = RGBColor(0xDC, 0xE6, 0xF1)
ROW_ALT = RGBColor(0xFA, 0xFA, 0xF8)

SW, SH = Inches(13.333), Inches(7.5)
FONT = "Segoe UI"

# (title, subtitle, figure stem or None, [bullets], speaker notes)
SLIDES: list[tuple] = [
    # ---------------- PERDE I ----------------
    ("__COVER__", "", None, [], ""),
    ("Bu çalışma neyi iddia ediyor, neyi iddia etmiyor",
     "Başarı tanımı en başta sabitlendi ve bir kere bile gevşetilmedi",
     None,
     ["İddia: İHA telemetrisinde hangi arıza ailesinde ve hangi veri koşulunda güvenilir alarm "
      "üretilebildiğini, hangilerinde üretilemediğini ölçülebilir biçimde gösterdik.",
      "İddia etmiyoruz: Her arıza tipini kapsayan genel bir UAV anomaly detector veya sahaya "
      "konabilecek dondurulmuş bir threshold.",
      "Bu sunumun tek kuralı: yakalama oranı ve yanlış alarm yükü hiçbir slaytta ayrı verilmez."],
     "Bu projede asıl risk düşük skor değildi; iyi görünen bir skorun gerçek zamanda taşınabilir alarm "
     "yüküne dönüşmemesiydi. Bu yüzden başarı tanımını en başta sabitledik. Bugün göreceğiniz her sayı "
     "bu tanıma göre üretildi. Sunumda sayıların düştüğü birkaç yer var — orada model kötüleşmedi, "
     "ölçüm dürüstleşti."),
    ("Aynı veriden üç farklı soru sorulabilir",
     "Window, event ve flight birbirinin yerine kullanılamaz",
     "F03_uc_olcum_duzeyi", [],
     "Proje boyunca en pahalı hatamız bu üçünü karıştırmaktı. Window, modelin skor ürettiği kısa bir "
     "telemetri dilimi. Event, ardışık alarm pencerelerinin birleşip gerçek arıza aralığıyla eşleşmesi. "
     "Flight ise sadece 'bu uçuşta en az bir alarm var mı' sorusuna cevap veriyor. Bir uçuşta alarm "
     "görülmesi, arızanın yakalandığı anlamına gelmiyor — sunumun sonunda bunun sayısal kanıtını "
     "göstereceğim. Operatör açısından anlamlı olan tek düzey ortadaki."),
    ("Geçerli alarm nedir ve bu problem neden zor",
     "Ölçüt çifti: event recall + false event / normal flight-hour",
     "F04_gecerli_alarm", [],
     "Zorluk üç yerden geliyor. Birincisi, arıza örneği az ve pahalı — gerçek uçakta arıza üretmek riskli, "
     "ALFA'nın rudder sınıfında sadece 4 uçuş var. İkincisi, normal davranış çok çeşitli; agresif bir "
     "manevra ile bir arıza telemetride benzer görünebiliyor. Üçüncüsü, aynı skor farklı uçuş fazında "
     "farklı anlama geliyor. Bu yüzden AUC'yi hiçbir zaman tek başına başarı saymadık; AUC yalnız "
     "sıralama gücü gösterir, alarm garantisi vermez."),
    ("Değerlendirme sözleşmesi: dört kural, en baştan dondurulmuş",
     "Her kural bir hatanın bedeli ödendikten sonra kurala dönüştü",
     "F05_degerlendirme_sozlesmesi", [],
     "Bu dört kural sonradan eklenmedi; her biri bir hatanın bedelini ödedikten sonra kurala dönüştü — "
     "birazdan dördünün de nasıl doğduğunu tek tek göreceksiniz. Şunu şimdiden söyleyeyim: bu kuralların "
     "her biri elimizdeki en iyi sayıyı düşürdü. Sunum boyunca göreceğiniz sayı düşüşlerinin çoğu model "
     "kötüleşmesi değil, ölçümün dürüstleşmesidir."),

    # ---------------- PERDE II ----------------
    ("Neden dışarıdan gözlem yetmedi",
     "Aynı uçuş, iki farklı gözlem katmanı",
     "F06_gozlem_katmani", [],
     "Projeye hava trafiği verisiyle başladık. Kısa sürede şunu gördük: dışarıdan gözlem aracın içinde ne "
     "olduğunu söylemiyor. Rota sapması ve irtifa değişimi görülebiliyor; motor gücünün düştüğü, bir "
     "sensörün bozulduğu ya da bunun GPS spoofing mi gerçek manevra mı olduğu görülemiyor. Bu veriyi "
     "bırakmadık — rolünü değiştirdik. Etiket üretemediği için 'arıza yakalama' verisi olmaktan çıktı, "
     "devasa gerçek normal trafik sunduğu için 'yanlış alarm stres testi' verisi oldu. Bu arada kimlik "
     "doğrulama ve kota kısıtları nedeniyle sağlayıcıyı da değiştirdik; çalışmanın tamamı adsb.lol verisi "
     "üzerinde yapıldı."),
    ("Beş veri kaynağı, beş ayrı teknik soru",
     "Beş ayrı sonuç üretmek için değil, problemin beş bileşenini izole etmek için",
     None,
     ["ADS-B (adsb.lol) — ~497.571 skorlanabilir normal flight-hour, etiket yok "
      "→ alarm bütçesi gerçek trafikte ne kadar hızlı bozulur?",
      "ALFA — sabit kanatlı gerçek uçuş, motor ve control-surface arıza aralıkları "
      "→ etiketli ama küçük veride ne öğrenilebilir?",
      "UAV Attack — Ping DoS, GPS spoofing, jamming senaryoları "
      "→ etiketin varlığı, ölçülebilir sinyalin varlığı mıdır?",
      "UAV-SEAD — 1.044 development uçuşu "
      "→ uçuş sayısı bağımsız deney sayısı mıdır?",
      "RflyMAD — 6.605 uçuş; Real / HIL / SIL; beş arıza ailesi "
      "→ domain farkı ve gerçek interval truth altında ne kalır?"],
     "Bu beş kaynak beş ayrı sonuç üretmek için seçilmedi. Her biri problemin başka bir bileşenini izole "
     "ediyor: etiket kalitesi, fiziksel gözlenebilirlik, veri bağımsızlığı, domain kayması ve yanlış alarm "
     "maruziyeti. Bir kaynağın zayıf olduğu yer, diğerinin test alanı oldu. Dürüst olmak gerekirse: her "
     "yöntemi her veri setinde aynı kapsamda çalıştırmadık; yöntemleri hedefli kullandık."),
    ("Ham kolonlardan feature'a: veri ilk hâliyle kullanılamıyordu",
     "ALFA örneği — model kurmadan önce yapılan üç müdahale",
     "F08_kolon_onarimi", [],
     "Buradaki ders şu: model kurmadan önce en az bir kez veriyi kolon kolon denetlemek gerekiyor. Boş "
     "görünen bir kolon çoğu zaman eksik veri değil, yanlış eşleştirme oluyor — velocity_mps kolonunda "
     "tam olarak bu oldu. Ve normal uçuş sayısını 10'dan 15'e çıkarmak küçük bir sayı gibi görünüyor ama "
     "normal-only bir modelde normal havuzunu yüzde elli büyütmek demek; birazdan bunun sonuca etkisini "
     "göreceksiniz."),
    ("Arıza 'motor_failure' adlı bir kolonda gelmiyor",
     "Feature tasarımı — arızanın uçuş dinamiğinde bıraktığı ikincil fiziksel iz",
     "F09_feature_zinciri", [],
     "Somut örnek: motor arızasında irtifa hemen düşmez. Önce throttle komutu yüksek kalırken airspeed "
     "azalır, sonra autopilot telafi ederken attitude davranışı değişir. Yani ham irtifa kanalına bakan bir "
     "model arızayı geç görür; komut ile tepki arasındaki farka bakan bir model erken görür. UAV Attack "
     "tarafında en değerli tek feature GPS speed residual oldu: GPS'in raporladığı hız ile ardışık "
     "konumlardan hesaplanan hızın farkı. Feature engineering'in amacı modele ezber verdirmek değil, "
     "arızanın izini görünür hale getirmektir."),

    # ---------------- PERDE III ----------------
    ("Neden normal-only öğrenme",
     "Etiket eğitime girmez; yalnız threshold kalibrasyonu ve değerlendirmede kullanılır",
     "F10_normal_only", [],
     "Bu bir tercih değil, veri yapısının dayattığı bir zorunluluktu. Normal veri büyütülebilir bir kaynak — "
     "daha çok uçuş, daha çok saat. Etiketli arıza verisi büyütülemez; gerçek uçakta arıza üretmek riskli "
     "ve pahalı. Bir de şu var: supervised bir model gördüğü arıza tipine aşırı uyum sağlar ve görmediğine "
     "kör kalır. Denetimli alternatifi körü körüne reddetmedik — denedik, iki slayt sonra sonucunu "
     "göstereceğim."),
    ("Tek modele her şeyi vermek sinyali seyreltiyor",
     "Isolation Forest — aynı model, iki farklı feature kurulumu",
     "F11_monolitik_moduler", [],
     "Isolation Forest'ı seçmemizin sebebi netti: etiket istemez, normal veriden aykırılık skoru üretir, "
     "hızlıdır ve hangi feature'ın katkı verdiği okunabilir. Ama monolitik kurulumda güçlü ama dar bir "
     "sinyal, alakasız feature'lar içinde seyreldi. Feature'ları fiziksel modüllere ayırdık — "
     "control-response, guidance, navigation, signal quality — her modülü ayrı skorladık, skorları normal "
     "validation verisinde ölçekleyip birleştirdik. Kazanım modelden gelmedi, model aynı Isolation Forest; "
     "kazanım sinyalin seyrelmesini önlemekten geldi."),
    ("Aynı model, daha çeşitli veri",
     "LSTM autoencoder — mimari hiç değişmedi, yalnız normal uçuş havuzu büyüdü",
     "F12_veri_cesitliligi", [],
     "Autoencoder'a geçmemizin sebebi Isolation Forest'ın zaman sırasını kullanmamasıydı. Autoencoder "
     "normal telemetri penceresini yeniden kurmayı öğrenir; kuramadığı yer anomali adayıdır. Eksik sensör "
     "hücrelerini sıfır hata saymadık, yalnız gözlenen değerler loss'a katkı verdi. Kolay yorum 'derin "
     "model bu problemde çalışmıyor' olurdu. Ama 5 normal uçuş eklediğimizde aynı mimari çok farklı bir "
     "sıralama sinyali üretti. Şunu şimdi söylemem lazım: bu 0,918 nihai bir sonuç değil, birkaç slayt "
     "sonra 0,611'e düşecek. Bu sayıyı başarı olarak sunmuyorum, sadece veri çeşitliliğinin etkisini "
     "gösteriyorum."),
    ("Modelin neye duyarlı olduğunu nasıl ölçtük",
     "Kontrollü bozulma enjeksiyonu — yalnız değerlendirmede, eğitime hiç girmedi",
     "F13_bozulma_taksonomisi", [],
     "Gerçek arıza örneği az olduğu için, 'model çalışmıyor' demeden önce hangi bozulma tipinde çalışıp "
     "hangisinde çalışmadığını ayırmamız gerekiyordu. Altı kontrollü bozulma tipi enjekte ettik. Sonuç "
     "şuydu: 'anomali' tek bir kategori değil. CUSUM giderek büyüyen sapmayı yaklaşık dörtte üç oranında "
     "yakaladı; buna karşılık sabit kayma ve sinsi konum kayması ölçtüğümüz bütün yöntemlerde en zor iki "
     "sınıf olarak kaldı. Bu enjeksiyonlar hiçbir zaman eğitime girmedi — girseydi model tam da o "
     "bozulmaları ezberlerdi."),
    ("Yakalama arttı; alarm yükü daha hızlı arttı",
     "UAV-SEAD üzerinde yöntem turu — her adım bir öncekinin bıraktığı boşluğu test etti",
     "F14_yontem_turu", [],
     "İki tanesini vurgulayayım. EKF innovation'ı denememizin sebebi, uçuş kontrolcüsünün zaten kendi "
     "tutarlılık göstergesini üretmesiydi — bedava bir sinyal gibi görünüyordu. Ters yönde sinyal verdi; "
     "sebebi şu: anomalili ölçüm estimator tarafından reddedildiğinde innovation düzenli üretilmiyor, yani "
     "en sorunlu örnekler daha temiz görünüyor. Bu bir bug değil, göstergenin doğası. İkincisi: tek bir "
     "feature, thrust command, en yüksek kategori yakalamasını verdi — ama saatte 38 yanlış alarmla. "
     "Soldan sağa yakalama gerçekten yükseliyor; her adımda alarm yükü daha hızlı yükseldi. Bu tablo "
     "'daha iyi model bulduk' demiyor, 'aynı duvara farklı yollardan çarptık' diyor."),
    ("Denetimli öğrenme neden ana hat olmadı",
     "Reddetmeden önce iki denetimli alternatif ölçüldü",
     "F15_denetimli_alternatifler", [],
     "LightGBM'i denedik çünkü etiketler elimizdeydi ve tablo verisinde gradient boosting en güçlü baseline "
     "sayılır. Isolation Forest'ın altında kaldı. TCN'i denedik çünkü temporal convolution uzun "
     "pencerelerde sıralı örüntüyü yakalamada güçlüdür ve rüzgâr kaynaklı yanlış alarmı bastırmasını "
     "umduk. En kritik bulgu şu: TCN'in en iyi durma noktası 2 ile 5 epoch arasındaydı, daha uzun eğitim "
     "sonucu kötüleştirdi. Bu eğitim yetersizliği değil, erken ezberleme işaretidir — model az sayıdaki "
     "arıza örneğini ezberliyor, yeni arızaya genellemiyor. Etiketli arıza verisi büyütülemediği için ana "
     "hat normal-only kaldı."),
    ("Etiketsiz devasa trafikte iki ders",
     "256.150.550 satır işlendi · ~497.571 skorlanabilir normal flight-hour",
     "F16_adsb_iki_ders", [],
     "İlk ders bu projenin dönüm noktasıydı. Sentetikte yüzde 97,6 gören bir sistemin gerçek trafikte "
     "saatte 25 alarm üretmesi, o sistemin kullanılamaz olduğu anlamına gelir. Yaklaşımı arşivledik ve "
     "raporlama kuralını değiştirdik: yakalama oranı bir daha asla alarm yükü olmadan raporlanmadı. İkinci "
     "ders daha ilginç: soruyu daralttık — 'bu uçuş anormal mi' yerine 'fiziksel olarak ilişkili iki "
     "sinyalin tutarlılığı bozuldu mu' diye sorduk. Beş basit tutarlılık kuralı sinir ağlarını geçti. Ama "
     "dürüst olayım: karşılaştırdığımız o üç sinir ağı zaten ayrı bir kontrolde elenmişti, birazdan "
     "anlatacağım genlik problemi yüzünden."),

    # ---------------- PERDE IV ----------------
    ("Düzeltme 1 — model geleceğe bakıyordu",
     "CUSUM: küçük ama kalıcı sapmanın kanıtını zaman içinde biriktirir",
     "F17_nedensellik", [],
     "CUSUM'u eklememizin sebebi netti: tek pencere eşiği küçük ama süren bir sapmayı kaçırıyordu; CUSUM "
     "bu kanıtı zaman içinde biriktiriyor. Yöntem doğruydu, kurulumunda hata vardı. Baseline'ı uçuşun "
     "tamamından hesaplayınca model, geçmişte karar verirken gelecekteki veriden etkileniyordu. Bu sinsi "
     "bir hata — kod çalışıyor, sonuç iyi görünüyor, ama gerçek zamanda üretilemeyecek bir skor "
     "üretiyorsunuz. Düzelttik, sayı 0,878'den 0,611'e düştü. Bu bir performans kaybı değil; o 0,878 "
     "hiçbir zaman gerçek değildi. Düzeltmeden sonraki gerçek en güçlü tekil sinyal cross-track error "
     "oldu, 0,751."),
    ("Düzeltme 2 — arızadan önce açılmış alarm yakalama değildir",
     "Eski değerlendirici yalnız çakışmaya bakıyordu",
     "F18_gercek_baslangic", [],
     "Operasyonel olarak düşünün: alarm zaten açıktı, arıza sonra başladı. Bu alarm arızayı haber vermedi, "
     "tesadüfen üstüne denk geldi. Ölçüm bunu yakalama sayarsa sistemin gerçek uyarı kabiliyetini üç kat "
     "fazla gösterir — nitekim öyle olmuş. Değerlendiriciyi değiştirdik: artık arıza başladıktan sonra "
     "yeni bir alarm başlangıcı arıyor. Bu düzeltmeden sonra elimizdeki bütün geçmiş event sonuçlarını "
     "geçersiz saydık ve yeniden hesapladık. Aynı ilke sonradan RflyMAD ve ADS-B değerlendirmelerinde de "
     "aynen kullanıldı."),
    ("Düzeltme 3 — uçuş sayısı bağımsız deney sayısı değil",
     "UAV-SEAD havuzu 60'tan 1.044 development uçuşuna büyütüldü — ama uçuşlar akrabaydı",
     "F19_oturum_split", [],
     "Buradaki iyileşme yeni bir modelden gelmedi, doğru veri bölünmesinden geldi — ve dikkat edin, bu "
     "sefer sayı yükseldi. Oynaklığın on yediye bir düşmesi asıl kritik olan: öncesinde aynı deneyi farklı "
     "rastgele tohumla çalıştırdığınızda çok farklı sonuç alıyordunuz, yani hiçbir sonuca güvenilemezdi. "
     "Bu dersi sonrasında bütün veri setlerine taşıdık: RflyMAD'de senaryo ve domain grupları, UAV "
     "Attack'ta kampanya ve platform grupları."),
    ("Düzeltme 4 — model örüntü değil, sinyal büyüklüğü öğreniyordu",
     "Projenin en önemli metodolojik bulgusu",
     "F20_genlik_baskinligi", [],
     "Üç farklı derin mimari neredeyse aynı sonucu verdi. Bu, mimarilerin iyi olmasından çok üçünün de "
     "aynı kestirme yolu bulmuş olmasına benziyordu. Test şu: eğitilmiş modelin skoru ile hiç eğitilmemiş "
     "rastgele bir modelin skorunu karşılaştır. Neredeyse mükemmel korele çıktı — yani ağırlıkların hiçbir "
     "katkısı yok. Model 'anomali' öğrenmemiş, sadece 'büyük değer' öğrenmiş. Agresif bir manevrada roll "
     "büyüktür, model onu anomali sanar. Doğrulamak için skoru genlikten arındırdık; genlik bağımlılığı "
     "düştü ama yakalama da çöktü — demek ki elimizdeki sinyal zaten büyüklükten ibaretti. O günden sonra "
     "hiçbir modeli bu kontrolden geçmeden rapor etmedik. Bu kapı, sunumun geri kalanındaki bütün "
     "sonuçların ön şartıdır."),
    ("Düzeltme 5 — modeli değil, sorulan soruyu düzelttik",
     "RflyMAD ground truth onarımı — modelde hiçbir değişiklik yapılmadı",
     "F21_truth_onarimi", [],
     "İki hata da modelin değil, ground truth'un hatasıydı. Birincisi ölçüm tanımından geliyordu: arızalı "
     "bir uçuşun tamamını anomali saymak, modelin uçuşun herhangi bir yerinde alarm vermesini yeterli "
     "kılıyor. İkincisi bir yazılım hatasıydı — 2.712 uçuşta arıza sanki ilk saniyeden başlıyor gibi "
     "görünüyordu. Bu hatadan etkilenen bütün geçmiş sonuçları geçersiz ilan ettik. Projedeki en büyük "
     "kazanımlardan biri yeni bir model değil, doğru bir ground truth oldu."),
    ("Dar çerçeve denemesi: yalnız GPS bütünlüğü",
     "Genel dedektör yerine tek bir arıza ailesine odaklanmak sonucu değiştirir mi?",
     "F22_gnss_pilotu", [],
     "Genel amaçlı bir dedektör her arıza tipinde ortalama başarı gösteriyordu. Soruyu daralttık: tek bir "
     "arıza ailesine odaklanırsak işletilebilir bir çalışma noktası bulunur mu? Rolleri ayırdık — 23 "
     "geliştirme, 15 prova, 20 mühürlü uçuş. Üç yöntemi de dondurulmuş bütçe altında ölçtük. Provada "
     "bağlamsal LSTM yüzde 90 yakalama ve sıfır alarm-saat gibi çok iyi göründü — ama bu geliştirmeye "
     "taşınmadı; eşik ve model roller arasında kararlı genellemedi. Üçü de aynı disiplinli sonuçla "
     "kapandı ve mühürlü set açılmadı."),
    ("Genlik problemine yapısal cevap: komut → tepki residual",
     "Anomaliyi ham büyüklükte değil, tahmin hatasında ara",
     "F23_komut_tepki_residual", [],
     "Genlik baskınlığı bulgusundan sonraki mantıklı adım buydu. Ham telemetri büyüklüğüne bakan bir skor "
     "her zaman büyük değerleri anomali sanacak. Onun yerine öğrenilmiş uçuş dinamiğinin tahmin hatasına "
     "bakıyoruz. Bilinen bir riski de baştan söyleyeyim: model, bir sonraki değeri bir öncekinden kopyalamayı "
     "öğrenip arızayı bir adım geriden takip edebilir. Bu yüzden tepki geçmişini sınırladık, komutu ve yavaş "
     "bağlamı öne çıkardık. Sonuç: arıza öncesi ve sonrası residual dağılımları eşikten bağımsız olarak "
     "ayrıştı — sinyal gerçekten var. Ama güvenilir bir eşik kurmak için bağımsız normal uçuş süresi "
     "yetersizdi. Eksik olan model değil, veri."),

    # ---------------- PERDE V ----------------
    ("Final yaklaşım: reconstruction yerine next-step forecasting",
     "Ortak protokol — dört veri setinde birebir aynı kurulum",
     "F24_final_model", [],
     "Reconstruction'dan forecasting'e geçmemizin sebebi doğrudan genlik problemi. Bir autoencoder girdiyi "
     "yeniden kurmaya çalışır ve büyük değerleri yeniden kurmak doğal olarak zordur — bu yüzden skor "
     "büyüklüğü takip eder. Forecasting farklı bir soru sorar: 'bir sonraki adım ne olmalı'. Agresif bir "
     "manevrada değerler büyük olabilir ama komuta uygun tepki varsa tahmin hatası küçük kalır. Şunu da net "
     "söyleyeyim: bu bir algoritma değil, bir protokoldür — asıl değeri, sonuçların dört veri seti arasında "
     "karşılaştırılabilir olmasını sağlaması."),
    ("Skor: masked Gaussian NLL",
     "Klasik reconstruction error'dan ayrıldığı yer burası",
     "F25_gaussian_nll", [],
     "Bu skor 'tahmin ne kadar yanlıştı' ile 'model bu bölgede ne kadar emindi' sorularını aynı anda "
     "cevaplıyor. Model çok eminse küçük bir sapma bile anlamlı anomali sinyali olabilir; doğal "
     "değişkenliğin yüksek olduğu bir bölgede aynı hata daha az anomalik sayılır. Bir teknik not: "
     "validation NLL'imiz negatif çıkıyor. Bu bir hata değil — kod sabit terimi eklemiyor ve sigma birden "
     "küçük olduğunda logaritması negatif oluyor. Daha negatif bir validation NLL yalnızca normal veriyi "
     "daha iyi açıklayan bir checkpoint demektir; anomali yakalama kanıtı değildir."),
    ("Bağımsızlık sözleşmesi: beş rol, sıfır kesişim",
     "RflyMAD rol kartı",
     "F26_rol_karti", [],
     "Genelleme iddiası, akraba uçuşlar rollere dağılmadan yapılamaz. Her veri setine kendi bağımsızlık "
     "birimini tanımladık; RflyMAD'de bu senaryo ve domain gruplarıydı. Son sütun önemli: 553 uçuşluk "
     "final test setini mühürledik ve hiç açmadık. Çünkü development sonucu geçme eşiğimizin altında "
     "kaldı — mühürlü seti açmak, sonuca bakıp karar vermek olurdu. Bu setin kapalı kalması bu projenin en "
     "savunulabilir kararıdır."),
    ("İlk kez genlik kontrolünden temiz geçen model",
     "Aynı teşhis, iki farklı tur — kapı: ρ < 0,80",
     "F27_genlik_kapisi", [],
     "Bu, projedeki tek gerçek 'iyileşme' hikâyesi. Daha önce gösterdiğim 0,964'ten buraya, 0,33'e geldik — "
     "model artık gerçekten örüntü öğreniyor, büyüklük kopyalamıyor. Checkpoint'i yalnız normal validation "
     "NLL ile seçtik, threshold'u validation tarafında dondurduk. Ama şunu net söylemem lazım: bu kapıyı "
     "geçmek 'model başarılı' demek değil. Yalnızca 'bu spesifik kestirme yolunu kullanmıyor' demek. "
     "Tespit performansı ayrı bir soru ve cevabını bir sonraki slaytta vereceğim."),
    ("Dört veri setinde ne çıktı",
     "Aynı protokol, dört farklı ders",
     "__TABLE_FOUR__", [],
     "Dört sonuç, dört farklı ders. RflyMAD tek koşullu 'devam' kararı — sıralama sinyali gerçekten var. "
     "UAV-SEAD çarpıcı: interval AUC 0,747, yani skor sıralama yapabiliyor; ama 206 gerçek olayın sadece "
     "7'sini bulmuş. Sıralama sinyalinin olay yakalamaya dönüşmediğinin en net kanıtı bu. ALFA'da sıfır "
     "yanlış alarm görüyorsunuz — bunu başarı diye sunmuyorum, çünkü bağımsız normal test maruziyetimiz "
     "sadece 0,131 saat; bu maruziyette sıfır alarm güçlü kanıt değil. UAV Attack ise zaten interval "
     "truth'u olmayan bir veri seti; oradaki tek anlamlı sayı saatte 32 yanlış olay."),
    ("ALFA'nın son sözü: grup-güvenli çapraz doğrulama",
     "Oturum-ailesi bazlı dört bağımsız grup — RflyMAD ile aynı sıkı sınav",
     "F29_alfa_group_cv", [],
     "RflyMAD gibi ALFA da en sonunda aynı sıkı sınavdan geçti. Dört gruptan yalnız biri genlik kapısını "
     "geçti — ama o grupta model hiçbir ayrım yapmadı: yakalama sıfır, normal alarm sıfır, eşik hiç "
     "aşılmadı. Diğer üç grup kapıda kaldı. En yüksek yakalamayı gösteren Grup 3 aynı zamanda normal "
     "uçuşların yüzde 83'ünde alarm üretti — yani ayrım değil, sürekli alarm. ALFA'da da nihai karar "
     "RflyMAD ile aynı. Bu sonuç geliştirme aşamasında kaldı, resmî bir final test sonucu değil."),
    ("Çalışma noktası: yakalama ve alarm yükü aynı yerde buluşmadı",
     "RflyMAD — 144 aday noktadan 54'ü ön-kayıtlı olarak değerlendirildi",
     "F30_calisma_noktasi", [],
     "Bu slayt sunumun en önemli sayısal sonucu. Threshold'u gevşetince daha fazla arıza yakalıyorsunuz — "
     "ama normal uçuşlardaki yanlış alarm çok daha hızlı artıyor. Operatör açısından saatte 9 yanlış olay "
     "demek, sistemin ilk gün kapatılması demek. Sıralama sinyali var, işletilebilir bir çalışma noktası "
     "yok. Ve bunu 54 farklı noktayı önceden kaydedip değerlendirerek gösterdik — sonuca bakıp en iyi "
     "noktayı seçmedik."),
    ("Kör noktalar: ortalama yanıltıyor",
     "Aynı model, aynı eşik — yakalamanın tamamı tek bir yerde toplanıyor",
     "F31_kor_noktalar", [],
     "Ortalama yüzde 43'lük yakalama tek bir yerde toplanıyor: simülasyondaki motor arızaları. Sistemin "
     "asıl hedefi olan gerçek uçuş verisinde yakalama yüzde 7,8 — ve yakaladığı olaylarda bile 41 saniye "
     "gecikiyor. Motor arızalarında gecikme 0,1 saniye. Yani model komut–tepki ilişkisinin bozulduğu ani "
     "arızaları görüyor, yavaş gelişen sensör bozulmalarını görmüyor. Bunu düzeltmeyi denedik: altı "
     "ön-kayıtlı dayanıklılık adayı taradık, gerçek uçuş yakalaması ikiye katlandı ama genel yakalama "
     "düştü. Belirli bir domain iyileşirken genel sistem kötüleşti."),
    ("Aynı model, iki farklı soru",
     "Tek bir metriğin bir sonucu nasıl tamamen ters gösterebileceğinin kanıtı",
     "F32_any_window_yanilsamasi", [],
     "Bu slaydı özellikle koydum. Eğer sunumu 'model anomali uçuşlarının yüzde 83'ünü yakaladı' diye "
     "bitirseydim, teknik olarak yanlış bir cümle kurmuş olmazdım. Ama aynı eşik normal uçuşların yüzde "
     "85'inde de alarm veriyor — yani model hiçbir ayrım yapmıyor. Dengeli doğruluk yüzde 49, tam olarak "
     "yazı-tura. Bu, tek bir metriğin bir sonucu nasıl tamamen ters gösterebileceğinin kanıtı ve bizim "
     "raporlama kuralımızın neden bu kadar katı olduğunun cevabı."),
    ("Tek uçuşta bir yakalanan olay",
     "Skor, dondurulmuş eşik ve gerçek arıza aralığı aynı zaman ekseninde",
     "__IMG_DETECTED__", [],
     "Bu, sistemin işe yaradığı senaryo: motor arızası. Skor arıza aralığı içinde eşiği aşıyor ve alarm "
     "aralığın içinde başlıyor. Bu grafik gerçek koşudan gelen ham araştırma çıktısıdır, temsilî bir çizim "
     "değil — eksen etiketleri İngilizce, o yüzden okuyarak geçiyorum."),
    ("Tek uçuşta bir kaçan olay",
     "En çok önemsediğimiz senaryo: gerçek uçuşta sensör arızası",
     "__IMG_MISSED__", [],
     "Sağdaki senaryo en çok önemsediğimiz durum — gerçek uçuşta sensör arızası — ve orada skor hiç "
     "yükselmiyor. Arıza boyunca eşiğin altında kalıyor; tek alarm arıza başlamadan önce üretildiği için "
     "yakalama sayılmıyor. Bu grafiği sunumdan çıkarabilirdim; çıkarmadım, çünkü asıl teknik sorunun "
     "nerede olduğunu en açık gösteren şey bu."),

    # ---------------- PERDE VI ----------------
    ("Karar ve sıradaki iş", "",
     "F34_karar", [],
     "Kararımız net: bu protokol hedeflenen güvenilirliği karşılamadı, bu yüzden mühürlü final test setini "
     "açmadık. Ama bu sonucun kendisi bir çıktı. Bugün şunları biliyoruz: alarm bütçesinin gerçek trafikte "
     "ne kadar hızlı bozulduğunu; bir modelin örüntü mü büyüklük mü öğrendiğini nasıl test edeceğimizi; "
     "uçuş sayısının bağımsız deney sayısı olmadığını; ve etiket zaman doğruluğunun model seçiminden daha "
     "belirleyici olduğunu. Sıradaki veri kampanyasının neye ihtiyaç duyduğunu artık tahmin etmiyoruz, "
     "ölçtük. Yanlış bir sistemi sahaya vermektense, iyi belgelenmiş bir olumsuz sonuç vermeyi tercih ettik."),
    ("__CLOSING__", "", None, [], ""),
]

FOUR_DATASET_TABLE = [
    ["Veri seti", "Interval ROC-AUC", "Event recall", "Yanlış olay / normal saat"],
    ["RflyMAD", "0,907", "%56,1", "0,768"],
    ["UAV-SEAD", "0,747", "%3,40  (206 olayın 7'si)", "0"],
    ["ALFA", "—", "~%20", "0  —  ama yalnız 0,131 saat maruziyet"],
    ["UAV Attack", "0,000  (uçuş)", "interval truth yok", "32,42"],
]

REFERENCES = [
    "Chandola, Banerjee & Kumar (2009) — Anomaly Detection: A Survey",
    "Pang, Shen, Cao & van den Hengel (2021) — Deep Learning for Anomaly Detection: A Review",
    "Liu, Ting & Zhou (2008) — Isolation Forest",
    "Page (1954) / sequential change detection — CUSUM literatürü",
    "UAV telemetry anomaly detection ve GNSS integrity literatürü",
    "Veri kaynakları:  adsb.lol  ·  ALFA  ·  UAV Attack  ·  UAV-SEAD  ·  RflyMAD",
]


def _tb(slide, l, t, w, h):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def _p(tf, text, size, color, bold=False, space_after=0, first=False, space_before=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.text = text
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    f = p.runs[0].font if p.runs else p.font
    f.size = Pt(size)
    f.color.rgb = color
    f.bold = bold
    f.name = FONT
    return p


def _white_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE


def _rule(slide, top):
    ln = slide.shapes.add_shape(1, Inches(0.62), top, Inches(12.1), Emu(9525))
    ln.fill.solid(); ln.fill.fore_color.rgb = RULE
    ln.line.fill.background()
    ln.shadow.inherit = False


def _header(slide, title, subtitle, idx, total):
    tf = _tb(slide, Inches(0.62), Inches(0.36), Inches(11.6), Inches(0.62))
    _p(tf, title, 27, NAVY, bold=True, first=True)
    top = Inches(1.02)
    if subtitle:
        tf2 = _tb(slide, Inches(0.63), Inches(1.02), Inches(11.6), Inches(0.34))
        _p(tf2, subtitle, 13.5, MUTED, first=True)
        top = Inches(1.44)
    _rule(slide, top)
    tf3 = _tb(slide, Inches(12.1), Inches(6.92), Inches(0.7), Inches(0.3))
    p = _p(tf3, f"{idx} / {total}", 10, GREY, first=True)
    p.alignment = PP_ALIGN.RIGHT
    return top


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    blank = prs.slide_layouts[6]
    total = len(SLIDES)

    for i, (title, subtitle, fig, bullets, notes) in enumerate(SLIDES, 1):
        s = prs.slides.add_slide(blank)
        _white_bg(s)

        if title == "__COVER__":
            bar = s.shapes.add_shape(1, Inches(0.9), Inches(2.42), Inches(1.5), Inches(0.06))
            bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
            bar.line.fill.background(); bar.shadow.inherit = False
            tf = _tb(s, Inches(0.9), Inches(2.75), Inches(11.5), Inches(2.2))
            _p(tf, "İHA Telemetrisinde Anomaly Detection", 40, NAVY, bold=True, first=True)
            _p(tf, "Sinyalden operasyonel alarma", 24, INK, space_before=8)
            _p(tf, "Beş veri kaynağı  ·  sekiz yöntem ailesi  ·  beş ölçüm düzeltmesi  ·  bir dürüst karar",
               14, MUTED, space_before=20)
            tf2 = _tb(s, Inches(0.9), Inches(6.5), Inches(8), Inches(0.4))
            _p(tf2, "YAPAY ZEKÂ YAZILIM TEKNOLOJİLERİ", 12, MUTED, bold=True, first=True)
            s.notes_slide.notes_text_frame.text = (
                "Açılış cümlesi: 'Bu sunum bir modelin nasıl çalıştığını değil, bir ölçüm sisteminin nasıl "
                "kurulduğunu ve neyi ölçmeyi başaramadığımızı anlatıyor.'")
            continue

        if title == "__CLOSING__":
            tf = _tb(s, Inches(0.9), Inches(0.75), Inches(11.5), Inches(0.6))
            _p(tf, "Referanslar", 27, NAVY, bold=True, first=True)
            _rule(s, Inches(1.42))
            tf2 = _tb(s, Inches(0.9), Inches(1.75), Inches(11.5), Inches(3.0))
            for j, r in enumerate(REFERENCES):
                _p(tf2, r, 13.5, INK if j < 5 else NAVY, bold=(j == 5),
                   space_after=9, first=(j == 0))
            tf3 = _tb(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(0.9))
            _p(tf3, "Dinlediğiniz için teşekkürler", 26, NAVY, bold=True, first=True)
            tf4 = _tb(s, Inches(0.9), Inches(6.5), Inches(8), Inches(0.4))
            _p(tf4, "YAPAY ZEKÂ YAZILIM TEKNOLOJİLERİ", 12, MUTED, bold=True, first=True)
            continue

        # Rendered figures already carry their own title, subtitle and rule at
        # 16:9 — placing a second pptx header would duplicate it and push the
        # image off the slide. Those slides are full-bleed; only the page
        # number is overlaid.
        if fig and fig not in ("__TABLE_FOUR__", "__IMG_DETECTED__", "__IMG_MISSED__"):
            s.shapes.add_picture(str(FIG / f"{fig}.png"), 0, 0, width=SW, height=SH)
            tf = _tb(s, Inches(12.1), Inches(6.98), Inches(0.7), Inches(0.3))
            p = _p(tf, f"{i} / {total}", 10, GREY, first=True)
            p.alignment = PP_ALIGN.RIGHT
            if notes:
                s.notes_slide.notes_text_frame.text = notes
            continue

        top = _header(s, title, subtitle, i, total)
        body_top = top + Inches(0.24)

        if fig == "__TABLE_FOUR__":
            rows, cols = len(FOUR_DATASET_TABLE), 4
            tbl_shape = s.shapes.add_table(rows, cols, Inches(0.9), body_top + Inches(0.35),
                                           Inches(11.5), Inches(3.1))
            tbl = tbl_shape.table
            tbl.columns[0].width = Inches(2.4)
            tbl.columns[1].width = Inches(2.5)
            tbl.columns[2].width = Inches(3.1)
            tbl.columns[3].width = Inches(3.5)
            for r, row in enumerate(FOUR_DATASET_TABLE):
                for c, val in enumerate(row):
                    cell = tbl.cell(r, c)
                    cell.text = val
                    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                    cell.margin_left = Inches(0.12)
                    para = cell.text_frame.paragraphs[0]
                    para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
                    f = para.runs[0].font if para.runs else para.font
                    f.name = FONT
                    f.size = Pt(14 if r else 12.5)
                    header = (r == 0)
                    highlight = (r == 1)
                    f.bold = header or highlight
                    f.color.rgb = WHITE if header else (NAVY if highlight else INK)
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = (
                        NAVY if header else TINT_BLUE if highlight else ROW_ALT)
            tf = _tb(s, Inches(0.9), Inches(5.9), Inches(11.5), Inches(0.9))
            _p(tf, "Bu dört sayı aynı değerlendirme turuna aittir ve kendi içinde tutarlıdır. RflyMAD için "
                   "birazdan görülecek grup-güvenli sonuç daha zorlu bir bölünmeye aittir — aradaki fark "
                   "model gerilemesi değildir.", 12, MUTED, first=True)
        elif fig in ("__IMG_DETECTED__", "__IMG_MISSED__"):
            src = EXISTING / ("S20_detected_motor_timeline.png" if fig == "__IMG_DETECTED__"
                              else "S20_missed_real_sensor_timeline.png")
            # artifact is 1.75:1 — size by the height left under the header
            avail_h = Inches(7.5) - body_top - Inches(0.62)
            pic_w = int(avail_h * 1.75)
            s.shapes.add_picture(str(src), int((SW - pic_w) / 2), body_top,
                                 width=pic_w, height=avail_h)
            tf = _tb(s, Inches(0.9), Inches(7.02), Inches(10.8), Inches(0.34))
            _p(tf, "Orijinal araştırma çıktısı — dondurulmuş eşik 10,0940. Eksen etiketleri üretim "
                   "script'inden geldiği için İngilizcedir.", 10.5, GREY, first=True)
        else:
            n = len(bullets)
            size = 17 if n <= 3 else 15
            gap = 34 if n <= 3 else 22
            hues = [BLUE, ORANGE, AQUA, VIOLET, RED]
            tf = _tb(s, Inches(0.95), body_top + Inches(0.55), Inches(11.4), Inches(4.9))
            for j, b in enumerate(bullets):
                head, _, tail = b.partition(" — ")
                p = _p(tf, "", size, INK, space_after=gap, first=(j == 0))
                if tail:
                    r1 = p.add_run(); r1.text = head + " — "
                    r1.font.bold = True; r1.font.color.rgb = hues[j % len(hues)]
                    r1.font.size = Pt(size); r1.font.name = FONT
                    r2 = p.add_run(); r2.text = tail
                    r2.font.color.rgb = INK; r2.font.size = Pt(size); r2.font.name = FONT
                else:
                    r1 = p.add_run(); r1.text = "•   " + b
                    r1.font.color.rgb = INK; r1.font.size = Pt(size); r1.font.name = FONT

        if notes:
            s.notes_slide.notes_text_frame.text = notes

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"{len(SLIDES)} slaytlık deck yazıldı -> {OUT}")


if __name__ == "__main__":
    build()
