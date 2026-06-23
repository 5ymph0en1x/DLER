# =====================================================================
#  DLER  ->  Telegram : carte de fin de telechargement (IMAGE designee)
# ---------------------------------------------------------------------
#  Genere une carte PNG (rendu Edge headless) avec les donnees techniques
#  et l'envoie via sendPhoto. Aucune dependance a installer (Edge + curl
#  sont fournis avec Windows 11).
#
#  DLER : Reglages -> Automation -> Command :
#    powershell -NoProfile -ExecutionPolicy Bypass -File "C:\DLER\dler_telegram.ps1"
#  Previsualiser sans envoyer (garde le PNG + ouvre-le) :
#    $env:DLER_TG_DRYRUN='1'; .\dler_telegram.ps1
# =====================================================================

$Token  = 'PUT_YOUR_BOT_TOKEN_HERE'
$ChatId  = 'PUT_YOUR_CHAT_ID_HERE'

# --- Contexte fourni par DLER ---
$rel   = if ($env:DLER_RELEASE) { $env:DLER_RELEASE } else { 'Inconnu' }
$ok    = ($env:DLER_SUCCESS  -eq '1')
$rep   = ($env:DLER_REPAIRED -eq '1')
$dest  = if ($env:DLER_EXTRACT_DIR) { $env:DLER_EXTRACT_DIR } else { $env:DLER_OUTPUT_DIR }
$files = 0;  [void][int]::TryParse($env:DLER_FILES_EXTRACTED, [ref]$files)
$bytes = 0L; [void][int64]::TryParse($env:DLER_DOWNLOAD_BYTES, [ref]$bytes)
$dur   = 0;  [void][int]::TryParse($env:DLER_DURATION_S, [ref]$dur)
$segD  = 0;  [void][int]::TryParse($env:DLER_SEGMENTS_DONE, [ref]$segD)
$segT  = 0;  [void][int]::TryParse($env:DLER_SEGMENTS_TOTAL, [ref]$segT)
$conns = 0;  [void][int]::TryParse($env:DLER_CONNECTIONS, [ref]$conns)
$procs = 1;  [void][int]::TryParse($env:DLER_NUM_PROCESSES, [ref]$procs)
$mbps  = 0.0; [void][double]::TryParse(($env:DLER_AVG_SPEED_MBPS -replace ',', '.'),
                 [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$mbps)

# --- Formatage ---
function Fmt-Size([int64]$b) {
  if ($b -ge 1GB) { '{0:N1} Go' -f ($b/1GB) } elseif ($b -ge 1MB) { '{0:N0} Mo' -f ($b/1MB) }
  elseif ($b -ge 1KB) { '{0:N0} Ko' -f ($b/1KB) } else { "$b o" }
}
function Fmt-Dur([int]$s) {
  if ($s -ge 3600) { '{0}h{1:D2}' -f [int]($s/3600), [int](($s%3600)/60) }
  elseif ($s -ge 60) { '{0}m{1:D2}' -f [int]($s/60), ($s%60) } else { "${s}s" }
}
function Enc([string]$s){ ($s -replace '&','&amp;') -replace '<','&lt;' -replace '>','&gt;' }

$size  = Fmt-Size $bytes
$durS  = Fmt-Dur $dur
$spd   = '{0:N0} Mo/s' -f $mbps
$gbps  = '{0:N1} Gbps' -f ($mbps*8/1000)
$pct   = if ($segT -gt 0) { [math]::Min(100, [int]($segD*100/$segT)) } else { if ($ok){100}else{0} }
$relE  = Enc $rel
$destE = Enc $dest
$procLabel = if ($procs -gt 1) { "$procs procs" } else { 'mono' }
$segTxt = if ($segT -gt 0) { '{0:N0} / {1:N0}' -f $segD, $segT } else { 'n/a' }

if    (-not $ok) { $acc='#ff453a'; $accDim='#3a1d1d'; $stat='&#201;CHEC';        $glyph='&#10005;' }
elseif ($rep)    { $acc='#ff9f0a'; $accDim='#3a2c12'; $stat='R&#201;PAR&#201;';  $glyph='&#9874;'  }
else             { $acc='#30d158'; $accDim='#15331f'; $stat='TERMIN&#201;';      $glyph='&#10003;' }
$now = Get-Date -Format 'dd/MM/yyyy  HH:mm'
$destBlock = if ($dest) {
  "<div style='display:flex;align-items:center;gap:9px;margin-top:20px;font-family:&quot;Cascadia Code&quot;,&quot;Consolas&quot;,monospace;font-size:14px;color:#79808c'><span style='color:$acc'>&#128194;</span><span>$destE</span></div>"
} else { '' }

# --- Carte HTML (rendue par Edge en PNG @2x) ---
$tile = {
  param($val,$lab,$mono=$true)
  $f = if ($mono) { "font-family:'Cascadia Code','Consolas',monospace" } else { '' }
  "<div style='background:#21262f;border:1px solid #2d333f;border-radius:14px;padding:16px 18px'>" +
  "<div style='font-size:26px;font-weight:600;color:#f2f4f7;$f'>$val</div>" +
  "<div style='font-size:14px;color:#8b929e;margin-top:4px;letter-spacing:.3px'>$lab</div></div>"
}
$html = @"
<!doctype html><html><head><meta charset='utf-8'></head>
<body style='margin:0;width:100vw;height:100vh;background:#0f1115;display:flex;align-items:center;justify-content:center;box-sizing:border-box'>
<div style="width:860px;background:#15181e;border-radius:24px;border:1px solid #262c36;
     box-sizing:border-box;padding:30px 34px;font-family:'Segoe UI Variable','Segoe UI',system-ui,sans-serif;
     border-top:5px solid $acc">
  <div style='display:flex;align-items:center;justify-content:space-between'>
    <div style='display:flex;align-items:center;gap:11px'>
      <div style='width:13px;height:13px;border-radius:50%;background:$acc'></div>
      <span style='font-size:21px;font-weight:700;color:#fff;letter-spacing:1.5px'>DLER</span>
    </div>
    <div style='display:flex;align-items:center;gap:8px;background:$accDim;color:$acc;
         font-size:15px;font-weight:600;padding:7px 15px;border-radius:999px;letter-spacing:.5px'>
      <span style='font-size:16px'>$glyph</span> $stat
    </div>
  </div>
  <div style='font-size:27px;font-weight:600;color:#f6f8fb;margin:20px 0 22px;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden'>$relE</div>
  <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:13px'>
    $(& $tile $size 'Taille')
    $(& $tile $spd 'Vitesse moyenne')
    $(& $tile $durS 'Dur&#233;e')
    $(& $tile $files 'Fichiers')
    $(& $tile "$conns" 'Connexions')
    $(& $tile $procLabel 'Multiprocess' $false)
  </div>
  <div style='background:#21262f;border:1px solid #2d333f;border-radius:14px;padding:15px 18px;margin-top:13px'>
    <div style='display:flex;justify-content:space-between;font-size:14px;color:#8b929e;margin-bottom:9px'>
      <span style='letter-spacing:.3px'>Segments v&#233;rifi&#233;s</span>
      <span style="font-family:'Cascadia Code','Consolas',monospace;color:#cdd2da">$segTxt &nbsp; &#183; &nbsp; $gbps</span>
    </div>
    <div style='height:9px;background:#2d333f;border-radius:999px;overflow:hidden'>
      <div style='height:100%;width:$pct%;background:$acc;border-radius:999px'></div>
    </div>
  </div>
  $destBlock
  <div style='display:flex;justify-content:space-between;align-items:center;margin-top:14px;
       padding-top:14px;border-top:1px solid #262c36;font-size:13px;color:#6a717c'>
    <span style='letter-spacing:.5px'>NZB Downloader</span><span>$now</span>
  </div>
</div></body></html>
"@

# --- Rendu PNG via Edge headless ---
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe" }
$htmlPath = Join-Path $env:TEMP 'dler_card.html'
$png      = Join-Path $env:TEMP 'dler_card.png'
if (Test-Path $png) { Remove-Item -LiteralPath $png -Force }
Set-Content -LiteralPath $htmlPath -Value $html -Encoding UTF8
$url = ([System.Uri]$htmlPath).AbsoluteUri
& $edge --headless=new --disable-gpu --hide-scrollbars --default-background-color=0f1115ff `
        --force-device-scale-factor=2 "--screenshot=$png" --window-size=920,580 $url 2>$null | Out-Null
# Wait until the PNG EXISTS *and is fully released* (no lock). Edge/antivirus
# can hold a brief lock on the freshly-written file; curl -F then fails to open
# it (exit 26). Poll by trying an exclusive read-open until it succeeds.
$tries = 0
while ($tries -lt 60) {
  if (Test-Path $png) {
    try { $fsx = [System.IO.File]::Open($png, 'Open', 'Read', 'None'); $fsx.Close(); break }
    catch { }
  }
  Start-Sleep -Milliseconds 150; $tries++
}

# --- Diagnostic (visible en interactif + journalise dans %TEMP%\dler_telegram.log) ---
$log = Join-Path $env:TEMP 'dler_telegram.log'
function Note($m) {
  $line = '{0}  {1}' -f (Get-Date -Format 'HH:mm:ss'), $m
  $line | Out-File -FilePath $log -Append -Encoding UTF8
  Write-Host $line
}
$pngOk = Test-Path $png
Note ("Edge present : {0}" -f (Test-Path $edge))
Note ("PNG rendu    : {0}  ({1} octets)" -f $pngOk, ([int64]((Get-Item $png -ErrorAction SilentlyContinue).Length)))

if ($env:DLER_TG_DRYRUN -eq '1') { Note "DRYRUN -> $png"; return }

# --- Envoi via curl ---
if ($pngOk) {
  $pngU = $png.Replace('\', '/')
  $cap  = "<b>$stat</b> &#183; $relE"
  # CRITICAL: text fields use --form-string (literal). With plain -F, curl treats a
  # value starting with '<' as "read from file" and one with '@' as "upload file",
  # so the HTML caption "<b>...</b>" made curl look for a file -> exit 26.
  # Only the photo uses -F (it genuinely IS a file upload).
  $a = @('-sS','-F',"photo=@$pngU;type=image/png",
         '--form-string',"chat_id=$ChatId",'--form-string','parse_mode=HTML',
         '--form-string',"caption=$cap","https://api.telegram.org/bot$Token/sendPhoto")
  $resp = (& curl.exe @a 2>&1 | Out-String).Trim()
  Note "sendPhoto exit=$LASTEXITCODE"
} else {
  $txt  = "$stat - $rel`n$size | $spd | $durS | $conns conns"
  $resp = (& curl.exe -sS "https://api.telegram.org/bot$Token/sendMessage" `
             --data-urlencode "chat_id=$ChatId" --data-urlencode "text=$txt" 2>&1 | Out-String).Trim()
  Note "sendMessage (repli texte) exit=$LASTEXITCODE"
}
Note "reponse Telegram: $resp"
if ($resp -match '"ok":true') { Note "=> OK : message livre dans Telegram" }
else { Note "=> ECHEC : voir la reponse ci-dessus" }
