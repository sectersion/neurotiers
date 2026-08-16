$Header = "$($PSStyle.Foreground.Cyan)=== Launch Streamlit Demo ===$($PSStyle.Reset)"

Write-Host $Header
& python -m streamlit run demo.py
if ($LASTEXITCODE -eq 0) {
    exit 0
} else {
    Write-Host "$($PSStyle.Foreground.Red)Demo terminated with error$($PSStyle.Reset)"
    exit 1
}
