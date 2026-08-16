$Header = "$($PSStyle.Foreground.Cyan)=== Install Dependencies ===$($PSStyle.Reset)"

Write-Host $Header
& pip install -e .
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)Install successful$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Install failed$($PSStyle.Reset)"
    exit 1
}
