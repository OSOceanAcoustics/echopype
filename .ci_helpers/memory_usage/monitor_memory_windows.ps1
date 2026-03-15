New-Item -ItemType Directory -Force -Path ci_monitor | Out-Null
"timestamp,mem_used_mb" | Out-File ci_monitor/memory_usage.csv -Encoding ascii

while ($true) {
    $ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

    $os = Get-CimInstance Win32_OperatingSystem
    $total = $os.TotalVisibleMemorySize / 1024
    $free = $os.FreePhysicalMemory / 1024
    $used = [math]::Round($total - $free, 2)

    "$ts,$used" | Out-File ci_monitor/memory_usage.csv -Append -Encoding ascii

    Start-Sleep -Seconds 5
}
