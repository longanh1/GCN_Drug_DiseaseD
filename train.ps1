# Train model và lưu kết quả vào AI_ENGINE/data/results/
# Run: .\train.ps1 [-dataset C-dataset] [-model base|fuzzy|both]
param(
    [string]$dataset = "C-dataset",
    [string]$model   = "fuzzy"
)

$root   = $PSScriptRoot
$python = "C:\GCN_DrugDisease\.venv\Scripts\python.exe"

if ($model -eq "base" -or $model -eq "both") {
    Write-Host "Training AMNTDDA (base) on $dataset ..." -ForegroundColor Cyan
    & $python "$root\AI_ENGINE\src\train_DDA_base.py" --dataset $dataset
    Write-Host "Done [base]. Results saved to AI_ENGINE/data/results/" -ForegroundColor Green
}

if ($model -eq "fuzzy" -or $model -eq "both") {
    Write-Host "Training AMNTDDA_Fuzzy on $dataset ..." -ForegroundColor Cyan
    & $python "$root\AI_ENGINE\src\train_DDA_fuzzy.py" --dataset $dataset
    Write-Host "Done [fuzzy]. Results saved to AI_ENGINE/data/results/" -ForegroundColor Green
}
