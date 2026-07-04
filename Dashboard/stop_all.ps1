# stop_all.ps1
# stop_all.bat tarafindan cagirilir. Ana bilesenleri, PENCERE BASLIGI
# yerine PROCESS KOMUT SATIRI ICERIGINE gore bulup kapatir -- boylece
# elle (Ctrl+C + yeniden yazarak) baslatilmis pencereler de yakalanir,
# sadece start_all.bat ile acilanlar degil.

Write-Host "[1/3] Python surecleri kapatiliyor (adsb_producer/dashboard_consumer/app)..."
$pyProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'adsb_producer\.py|dashboard_consumer\.py|app\.py' }

if ($pyProcs) {
    foreach ($p in $pyProcs) {
        Write-Host "  kapatiliyor: PID=$($p.ProcessId)  $($p.CommandLine)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "  (calisan bulunamadi)"
}

Write-Host "[2/3] Redis / InfluxDB kapatiliyor..."
Stop-Process -Name "redis-server" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "influxd" -Force -ErrorAction SilentlyContinue

Write-Host "[3/3] Zookeeper + Kafka (Java) kapatiliyor..."
Get-Process -Name "*java*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Tamamlandi."
