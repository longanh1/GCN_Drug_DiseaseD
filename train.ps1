# Train model và lưu kết quả vào AI_ENGINE/data/results/
# Run: .\train.ps1 [-dataset C-dataset] [-model base|fuzzy|ablation|both]
param(
    [string]$dataset     = "C-dataset",
    [string]$model       = "fuzzy",
    [string]$variants    = "all",
    [int]$epochs         = 1000,
    [int]$num_threads    = 4,    # CPU threads PyTorch uses (lower = less heat)
    [int]$eval_every     = 1,    # evaluate every epoch
    [int]$patience       = 9999  # effectively disabled (no early stop)
)

$root   = $PSScriptRoot
$venv   = "$root\.venv"
$python = "$venv\Scripts\python.exe"
$pip    = "$venv\Scripts\pip.exe"

# ── Locate system Python ──────────────────────────────────────────────
$syspython = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $syspython) {
    Write-Error "Python not found on system PATH. Install Python 3.10+ first."
    exit 1
}

# ── Check if venv already has working dgl + torch ────────────────────
$dgl_ok = $false
if (Test-Path $python) {
    $dgl_ok = (& $python -c "import dgl, torch; print('ok')" 2>$null) -eq 'ok'
}

if (-not $dgl_ok) {
    # Remove broken venv and recreate
    if (Test-Path $venv) {
        Write-Host "Removing broken virtual environment ..." -ForegroundColor Yellow
        Remove-Item $venv -Recurse -Force
    }

    Write-Host "Creating virtual environment ..." -ForegroundColor Yellow
    & $syspython -m venv $venv
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv"; exit 1 }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green

    Write-Host "Installing PyTorch 2.0.1 (CPU) ..." -ForegroundColor DarkYellow
    & $pip install -q --upgrade pip setuptools wheel
    & $pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 `
        --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install PyTorch"; exit 1 }

    Write-Host "Installing DGL 1.1.2 ..." -ForegroundColor DarkYellow
    & $pip install -q dgl==1.1.2
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install DGL"; exit 1 }

    Write-Host "Installing remaining requirements ..." -ForegroundColor DarkYellow
    # Skip torch/dgl lines — already installed above
    $reqs = Get-Content "$root\AI_ENGINE\requirements.txt" |
        Where-Object { $_ -notmatch '^\s*(torch|dgl)' -and
                       $_ -notmatch '^\s*#' -and
                       $_.Trim() -ne '' }
    foreach ($req in $reqs) {
        & $pip install -q $req
    }

    # Verify
    $dgl_ok = (& $python -c "import dgl, torch; print('ok')" 2>$null) -eq 'ok'
    if (-not $dgl_ok) {
        Write-Error "DGL/PyTorch still not importable after install. Check output above."
        exit 1
    }
    Write-Host "✓ All dependencies ready (torch + dgl verified)" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment OK (torch + dgl already installed)" -ForegroundColor Green
}


if ($model -eq "base" -or $model -eq "both") {
    Write-Host "Training AMNTDDA (base) on $dataset ..." -ForegroundColor Cyan
    & $python "$root\AI_ENGINE\src\train_DDA_base.py" `
        --dataset $dataset --epochs $epochs `
        --num_threads $num_threads --eval_every $eval_every --patience $patience
    Write-Host "Done [base]. Results saved to AI_ENGINE/data/results/" -ForegroundColor Green
}

if ($model -eq "fuzzy" -or $model -eq "both") {
    Write-Host "Training AMNTDDA_Fuzzy on $dataset ..." -ForegroundColor Cyan
    & $python "$root\AI_ENGINE\src\train_DDA_fuzzy.py" `
        --dataset $dataset --epochs $epochs `
        --num_threads $num_threads --eval_every $eval_every --patience $patience
    Write-Host "Done [fuzzy]. Results saved to AI_ENGINE/data/results/" -ForegroundColor Green
}

if ($model -eq "gcn") {
    Write-Host "Training AMNTDDA_GCN on $dataset ..." -ForegroundColor Cyan
    & $python "$root\AI_ENGINE\src\train_DDA_gcn.py" `
        --dataset $dataset --epochs $epochs `
        --num_threads $num_threads --eval_every $eval_every --patience $patience
    Write-Host "Done [gcn]. Results saved to AI_ENGINE/data/results/" -ForegroundColor Green
}

if ($model -eq "ablation") {
    Write-Host "Training Ablation Study ($variants) on $dataset ..." -ForegroundColor Cyan
    & $python "$root\AI_ENGINE\src\train_DDA_ablation.py" `
        --dataset $dataset --variants $variants --epochs $epochs
    Write-Host "Done [ablation]. Results saved to AI_ENGINE/data/results/" -ForegroundColor Green
}
