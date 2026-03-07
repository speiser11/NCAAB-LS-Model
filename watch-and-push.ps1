# watch-and-push.ps1
# Watches index.html for changes and auto-commits + pushes to GitHub Pages.
# Run from a PowerShell terminal: .\watch-and-push.ps1
# Leave the window open while editing. Changes go live in ~60s after push.

$repoDir  = $PSScriptRoot
$watchFile = Join-Path $repoDir "index.html"
$lastWrite = (Get-Item $watchFile).LastWriteTime

Write-Host ""
Write-Host "  TrueLine auto-push watcher" -ForegroundColor Cyan
Write-Host "  Watching: $watchFile" -ForegroundColor Gray
Write-Host "  Save index.html to trigger commit + push -> GitHub Pages" -ForegroundColor Gray
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

while ($true) {
    Start-Sleep -Seconds 2
    $current = (Get-Item $watchFile -ErrorAction SilentlyContinue)?.LastWriteTime
    if ($current -and $current -ne $lastWrite) {
        $lastWrite = $current
        # Brief pause so the editor finishes writing the file
        Start-Sleep -Seconds 2
        $ts = (Get-Date).ToString('HH:mm:ss')
        Write-Host "[$ts] Change detected - staging..." -ForegroundColor Yellow
        Push-Location $repoDir
        git add index.html
        $dirty = git status --porcelain
        if ($dirty) {
            $msg = "Auto-save $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
            git commit -m $msg 2>&1 | Out-Null
            $push = git push origin main 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[$ts] Pushed - live on GitHub Pages in ~60s" -ForegroundColor Green
            } else {
                Write-Host "[$ts] Push failed: $push" -ForegroundColor Red
            }
        } else {
            Write-Host "[$ts] No staged changes (file saved but content unchanged)" -ForegroundColor Gray
        }
        Pop-Location
    }
}
