$Header = "$($PSStyle.Foreground.Cyan)=== Generate Benchmark Report ===$($PSStyle.Reset)"

Write-Host $Header
& python -m neuron_benchmark --report
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)Report generated$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Report generation failed$($PSStyle.Reset)"
    exit 1
}
