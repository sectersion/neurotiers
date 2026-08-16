$Header = "$($PSStyle.Foreground.Cyan)=== Run Tests ===$($PSStyle.Reset)"

Write-Host $Header
& python -m pytest
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)Tests passed$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Tests failed$($PSStyle.Reset)"
    exit 1
}
