@echo off
REM start_silver_realtime_loop.bat -- oturum acilinca Task Scheduler tarafindan
REM calistirilir (bkz. 2026-07-09 karari: gunluk realtime->Silver paketleme
REM dongusunun reboot sonrasi ELLE baslatilmasi gerekmesin diye eklendi).
REM Docker Desktop'in (ve icindeki MinIO/Kafka/InfluxDB container'larinin)
REM acilista hazir olmasi icin kisa bir bekleme birakiyoruz -- ilk birkac
REM deneme MinIO'ya erisemezse loop bunu zaten sessizce loglayip bir sonraki
REM (86400sn sonraki) denemede devam eder, veri kaybi olmaz (Bronze birikir).
REM 2026-07-09: --interval 86400 yerine --daily-at 17:00 -- PC sadece mesai
REM saatlerinde acik oldugu icin sabit sayac (PC kapaliyken donuyor) yerine
REM saat-bazli hedef kullaniyoruz (bkz. parse_adsblol_realtime.py --daily-at
REM aciklamasi). Giriste hemen bir catch-up + her gun 17:00'de calisir.
timeout /t 60 /nobreak >nul
cd /d "c:\Users\PC_6276\Desktop\github\AnomalyDetection"
"C:\Users\PC_6276\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m src.silver.parse_adsblol_realtime --daily-at 17:00 >> "c:\Users\PC_6276\Desktop\github\AnomalyDetection\data\state\silver_realtime_loop.log" 2>&1
