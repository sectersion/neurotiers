$Header = "$($PSStyle.Foreground.Cyan)=== Robustness Analysis ===$($PSStyle.Reset)"

Write-Host $Header
& python -m neuron_benchmark --robustness
if ($LASTEXITCODE -eq 0) {
    Write-Host "$($PSStyle.Foreground.Green)Robustness analysis completed$($PSStyle.Reset)"
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Robustness analysis failed$($PSStyle.Reset)"
    exit 1
}
