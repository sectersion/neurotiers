$Header = "$($PSStyle.Foreground.Cyan)=== Quick Smoke Test ===$($PSStyle.Reset)"

Write-Host $Header
& python -m neuron_benchmark --quick --epochs 3
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)Quick test passed$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Quick test failed$($PSStyle.Reset)"
    exit 1
}
