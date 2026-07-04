@echo off
echo ============================================
echo   TUM SISTEM KAPATILIYOR
echo ============================================
echo.

REM ONEMLI: eskiden pencere BASLIGINA gore ariyorduk (WINDOWTITLE eq
REM "ADS-B Producer" vb.) -- bu SADECE start_all.bat'in "start "Baslik"
REM ..." ile actigi pencerelerde calisir. Bu proje boyunca cogu zaman
REM adsb_producer.py / dashboard_consumer.py ELLE (mevcut bir pencerede
REM Ctrl+C + yeniden yazarak) yeniden baslatildi -- o pencerelerin
REM basligi "ADS-B Producer" DEGIL, bu yuzden eski stop_all.bat onlari
REM GOREMIYOR/KAPATAMIYORDU.
REM
REM COZUM: stop_all.ps1 -- pencere basligi yerine process'in KOMUT
REM SATIRI ICERIGINE gore arar (Get-CimInstance ile) -- nasil
REM baslatilmis olursa olsun (elle veya start_all.bat ile) buluyor.
REM Bu mantigi .bat icine gomulu PowerShell tek-satirlari olarak DEGIL,
REM ayri bir .ps1 dosyasi olarak yazdik -- ic ice tirnaklarin .bat
REM icinde bozulma riski yuksekti (daha once curl+JSON'da basimiza
REM gelmisti), ayri dosya cok daha guvenilir.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_all.ps1"

REM Eski pencere-basligi yontemi de zararsizca, ekstra guvence olarak
REM dursun -- start_all.bat ile acilmis pencereler icin hala isliyor,
REM yukaridaki komut satiri eslestirmesi zaten esas isi yapiyor.
taskkill /FI "WINDOWTITLE eq Redis*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq InfluxDB*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ADS-B Producer*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Dashboard Consumer*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Dashboard App*" /T /F >nul 2>&1

echo.
echo ============================================
echo   TAMAMLANDI
echo ============================================
echo Hala acik kalan bir pencere varsa (orn. check_redis_flights.py gibi
echo yardimci scriptler), onu elle kapatman gerekebilir -- bu script
echo sadece ana 5 servisi (Redis, InfluxDB, Kafka/Zookeeper, producer,
echo consumer, dashboard) hedefliyor.
pause
