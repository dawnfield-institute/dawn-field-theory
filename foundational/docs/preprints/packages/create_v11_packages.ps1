# create_v11_packages.ps1
# Creates v1.1 (or v2.1 for infodynamics) Zenodo packages for 6 updated papers.
# Run from: foundational/docs/preprints/packages/

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$srcBase = Split-Path $PSScriptRoot -Parent  # foundational/docs/preprints

# Paper definitions: slug, new_version, zenodo_version_on_record
$papers = @(
    @{ slug = "symbolic_entropy_collapse";       newVer = "1.1"; zenVer = "1.0" },
    @{ slug = "qbe_pac_unification";             newVer = "1.1"; zenVer = "1.0" },
    @{ slug = "dawn_field_theory_infodynamics";   newVer = "2.1"; zenVer = "2.0" },
    @{ slug = "ml_validation_pythia_gpt2";        newVer = "1.1"; zenVer = "1.0" },
    @{ slug = "cellular_automata_xi_clustering";  newVer = "1.1"; zenVer = "1.0" },
    @{ slug = "pac_necessity_proof";              newVer = "1.1"; zenVer = "1.0" }
)

foreach ($paper in $papers) {
    $slug = $paper.slug
    $ver = $paper.newVer
    $pkgName = "${slug}_v${ver}_${timestamp}"
    $srcDir = Join-Path $srcBase $slug
    $pkgDir = Join-Path $PSScriptRoot "$slug\$pkgName"
    $zipPath = Join-Path $PSScriptRoot "$slug\$pkgName.zip"

    Write-Host "`n=== Packaging $slug v$ver ===" -ForegroundColor Cyan

    # 1. Create package directory
    New-Item -ItemType Directory -Path $pkgDir -Force | Out-Null

    # 2. Copy source files (excluding .git, __pycache__, etc.)
    $excludePatterns = @("__pycache__", ".git", "*.pyc", ".DS_Store")
    Get-ChildItem $srcDir -Recurse -Force | Where-Object {
        $rel = $_.FullName.Substring($srcDir.Length + 1)
        $skip = $false
        foreach ($pat in $excludePatterns) {
            if ($rel -like "*$pat*") { $skip = $true; break }
        }
        -not $skip
    } | ForEach-Object {
        $relPath = $_.FullName.Substring($srcDir.Length + 1)
        $dest = Join-Path $pkgDir $relPath
        if ($_.PSIsContainer) {
            New-Item -ItemType Directory -Path $dest -Force | Out-Null
        } else {
            $destDir = Split-Path $dest -Parent
            if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
            Copy-Item $_.FullName $dest
        }
    }

    # 3. Copy .zenodo.json from previous package (if exists), update version
    $prevPkgDir = Get-ChildItem (Join-Path $PSScriptRoot $slug) -Directory | 
        Where-Object { $_.Name -ne $pkgName } | Sort-Object Name | Select-Object -Last 1
    if ($prevPkgDir) {
        $prevZenodo = Join-Path $prevPkgDir.FullName ".zenodo.json"
        if (Test-Path $prevZenodo) {
            $zenodoJson = Get-Content $prevZenodo -Raw | ConvertFrom-Json
            $zenodoJson.version = $ver
            $zenodoJson | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $pkgDir ".zenodo.json") -Encoding UTF8
            Write-Host "  .zenodo.json: copied and updated version to $ver"
        }
    }

    # 4. Update meta.yaml version
    $metaPath = Join-Path $pkgDir "meta.yaml"
    if (Test-Path $metaPath) {
        $content = Get-Content $metaPath -Raw
        $content = $content -replace '(?m)^version:\s*"[^"]*"', "version: `"$ver`""
        Set-Content $metaPath $content -NoNewline -Encoding UTF8
        Write-Host "  meta.yaml: version set to $ver"
    }

    # 5. Fix qbe_pac_unification paper.md header if still at v1.0
    if ($slug -eq "qbe_pac_unification") {
        $paperPath = Join-Path $pkgDir "paper.md"
        $paperContent = Get-Content $paperPath -Raw
        if ($paperContent -match '\*\*Version:\*\*\s*v1\.0') {
            $paperContent = $paperContent -replace '(\*\*Version:\*\*)\s*v1\.0', '$1 v1.1'
            Set-Content $paperPath $paperContent -NoNewline -Encoding UTF8
            Write-Host "  paper.md: version bumped to v1.1"
        }
    }

    # 6. Generate MANIFEST.json
    $manifest = @{
        created = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.ffffff")
        version = $ver
        files = @()
    }
    Get-ChildItem $pkgDir -Recurse -File | ForEach-Object {
        $relPath = $_.FullName.Substring($pkgDir.Length + 1)
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
        $manifest.files += @{
            path = $relPath
            size = $_.Length
            sha256 = $hash
        }
    }
    $manifest | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $pkgDir "MANIFEST.json") -Encoding UTF8
    Write-Host "  MANIFEST.json: $($manifest.files.Count) files indexed"

    # 7. Create zip
    if (Test-Path $zipPath) { Remove-Item $zipPath }
    Compress-Archive -Path "$pkgDir\*" -DestinationPath $zipPath
    $zipSize = (Get-Item $zipPath).Length
    Write-Host "  ZIP: $pkgName.zip ($([math]::Round($zipSize/1KB, 1)) KB)" -ForegroundColor Green
}

Write-Host "`n=== All packages created ===" -ForegroundColor Green
Write-Host "Timestamp: $timestamp"
