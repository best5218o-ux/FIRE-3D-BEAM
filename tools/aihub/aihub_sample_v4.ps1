# AI허브 239 - STR/SPA 평면도(FP) 표본 v4 (원본 zip 읽기만)
# 원본 zip마다 여러 묶음(각 330MB 이하), 파일명에 _FP_ 포함된 것만
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = Split-Path -Parent $PSScriptRoot
$out  = $PSScriptRoot
$cap  = 330MB
$parts = @{ 'TS_STR_1' = 3; 'TS_STR_2' = 3; 'TS_SPA_1' = 1; 'TS_SPA_2' = 1 }
$log  = Join-Path $out 'sample_log_v4.txt'
$lock = Join-Path $out 'sample_v4.lock'
if (Test-Path $lock) { "already running" | Out-File $log -Append -Encoding utf8; exit }
"run" | Out-File $lock -Encoding utf8
"start $(Get-Date)" | Out-File $log -Encoding utf8
try {
  $zips = Get-ChildItem -Path $root -Recurse -Filter *.zip | Where-Object { $_.Length -gt 0 -and $parts.ContainsKey($_.BaseName) }
  foreach ($zf in $zips) {
    try {
      $P = $parts[$zf.BaseName]
      $z = [System.IO.Compression.ZipFile]::OpenRead($zf.FullName)
      $ents = @($z.Entries | Where-Object { $_.Length -gt 0 -and $_.Name -like '*_FP_*' } | Sort-Object FullName)
      $avg = [double](($ents | Measure-Object -Property CompressedLength -Sum).Sum) / [math]::Max(1, $ents.Count)
      $per = [int]([double]$cap / [math]::Max(1, $avg) * 0.92)
      $target = [math]::Min($ents.Count, $per * $P)
      $stride = [math]::Max(1, [int][math]::Floor($ents.Count / [math]::Max(1, $target)))
      "$($zf.Name) FP=$($ents.Count) avgComp=$([int]$avg) per=$per parts=$P stride=$stride" | Out-File $log -Append -Encoding utf8
      $idx = @(); for ($k = 0; $k -lt $ents.Count; $k += $stride) { $idx += $k }
      $p = 1; $i = 0
      while ($p -le $P -and $i -lt $idx.Count) {
        $dst = Join-Path $out ("sample_" + $zf.BaseName + "_p" + $p + ".zip")
        if (Test-Path $dst) { Remove-Item $dst -Force }
        $fs = [System.IO.File]::Open($dst, 'CreateNew')
        $zo = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create, $true)
        $n = 0
        while ($i -lt $idx.Count -and $fs.Length -lt $cap) {
          $e = $ents[$idx[$i]]; $i++
          $ne = $zo.CreateEntry($e.Name, [System.IO.Compression.CompressionLevel]::Optimal)
          $s = $e.Open(); $d = $ne.Open(); $s.CopyTo($d); $d.Close(); $s.Close(); $n++
        }
        $zo.Dispose(); $fs.Close()
        "  part $p -> files=$n bytes=$((Get-Item $dst).Length)" | Out-File $log -Append -Encoding utf8
        $p++
      }
      $z.Dispose()
    } catch {
      "ERROR $($zf.FullName): $($_.Exception.Message)" | Out-File $log -Append -Encoding utf8
    }
  }
} finally {
  Remove-Item $lock -Force
  "done $(Get-Date)" | Out-File $log -Append -Encoding utf8
}
