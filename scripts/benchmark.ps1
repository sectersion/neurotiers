$Header = "$($PSStyle.Foreground.Cyan)=== Full Benchmark Run ===$($PSStyle.Reset)"

Write-Host $Header
& python -m neuron_benchmark --experiment --seeds 0 1 2 3 4 --epochs 20
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)Benchmark completed successfully$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Benchmark failed$($PSStyle.Reset)"
    exit 1
}
