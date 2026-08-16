$Header = "$($PSStyle.Foreground.Cyan)=== MNIST Multi-Seed Benchmark ===$($PSStyle.Reset)"

Write-Host $Header
Write-Host "$($PSStyle.Foreground.Yellow)Running multi-seed experiments on MNIST for all neurons...$($PSStyle.Reset)"

& python -m neuron_benchmark --experiment --dataset mnist --seeds 0 1 2 3 4 --epochs 20
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)MNIST benchmark completed successfully$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)MNIST benchmark failed$($PSStyle.Reset)"
    exit 1
}
