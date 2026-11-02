param(
    [string]$OldData = $PSScriptRoot,
    [string]$NewData = (Join-Path $PSScriptRoot 'dataset_clean'),
    [string]$Output = (Join-Path $PSScriptRoot 'dataset_merged')
)

$ErrorActionPreference = 'Stop'
$splits = @('train', 'valid', 'test')
$hashLocations = @{}
$boxes = @{ '0' = 0; '1' = 0 }
$summary = [System.Collections.Generic.List[object]]::new()
$duplicates = 0
$labelConflicts = [System.Collections.Generic.List[string]]::new()

if (Test-Path -LiteralPath $Output) { throw "Output already exists: $Output" }
foreach ($root in @($OldData, $NewData)) {
    foreach ($split in $splits) {
        foreach ($folder in @('images', 'labels')) {
            $path = Join-Path (Join-Path $root $split) $folder
            if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Missing folder: $path" }
        }
    }
}

function Read-CleanLabel([string]$Path) {
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line.Trim() -split '\s+'
        if ($parts.Count -ne 5) { throw "Label must contain five columns: $Path" }
        $classId = 0
        if (-not [int]::TryParse($parts[0], [ref]$classId) -or $classId -notin 0, 1) { throw "Invalid class ID: $Path" }
        $values = [System.Collections.Generic.List[double]]::new()
        foreach ($text in $parts[1..4]) {
            $value = 0.0
            if (-not [double]::TryParse($text, [Globalization.NumberStyles]::Float, [cultureinfo]::InvariantCulture, [ref]$value) -or [double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0 -or $value -gt 1) { throw "Invalid normalized coordinate: $Path" }
            $values.Add($value)
        }
        if ($values[2] -le 0 -or $values[3] -le 0) { throw "Zero-sized box: $Path" }
        $lines.Add(($classId.ToString() + ' ' + ($parts[1..4] -join ' ')))
    }
    if ($lines.Count -eq 0) { throw "Empty label: $Path" }
    return ,$lines
}

New-Item -ItemType Directory -Path $Output | Out-Null
try {
    foreach ($split in $splits) {
        $targetImages = Join-Path (Join-Path $Output $split) 'images'
        $targetLabels = Join-Path (Join-Path $Output $split) 'labels'
        New-Item -ItemType Directory -Path $targetImages, $targetLabels | Out-Null
        $kept = 0
        $seenNames = @{}

        # Prefer v2 files if the same image exists in both exports.
        foreach ($sourceRoot in @($NewData, $OldData)) {
            $imageDir = Join-Path (Join-Path $sourceRoot $split) 'images'
            $labelDir = Join-Path (Join-Path $sourceRoot $split) 'labels'
            foreach ($image in Get-ChildItem -LiteralPath $imageDir -File | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' }) {
                $labelPath = Join-Path $labelDir ($image.BaseName + '.txt')
                if (-not (Test-Path -LiteralPath $labelPath -PathType Leaf)) { throw "Missing label: $labelPath" }
                $cleanLabel = Read-CleanLabel $labelPath
                $hash = (Get-FileHash -LiteralPath $image.FullName -Algorithm SHA256).Hash
                if ($hashLocations.ContainsKey($hash)) {
                    if ($hashLocations[$hash] -ne $split) { throw "Duplicate image appears in different splits: $($image.Name)" }
                    $duplicates++
                    $existingImage = Get-ChildItem -LiteralPath $targetImages -File | Where-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -eq $hash } | Select-Object -First 1
                    if ($existingImage) {
                        $existingLabel = Join-Path $targetLabels ($existingImage.BaseName + '.txt')
                        $existingText = (Get-Content -LiteralPath $existingLabel) -join "`n"
                        $incomingText = $cleanLabel -join "`n"
                        if ($existingText -ne $incomingText) { $labelConflicts.Add("$split/$($image.Name)") }
                    }
                    continue
                }
                if ($seenNames.ContainsKey($image.Name)) { throw "Different images share a filename: $split/$($image.Name)" }
                $hashLocations[$hash] = $split
                $seenNames[$image.Name] = $true
                Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $targetImages $image.Name)
                [System.IO.File]::WriteAllLines((Join-Path $targetLabels ($image.BaseName + '.txt')), $cleanLabel, [System.Text.UTF8Encoding]::new($false))
                foreach ($line in $cleanLabel) { $classId = ($line -split ' ')[0]; $boxes[$classId]++ }
                $kept++
            }
        }
        $summary.Add([pscustomobject]@{ split = $split; image_count = $kept; label_count = $kept })
    }
    $yaml = @('path: .', 'train: train/images', 'val: valid/images', 'test: test/images', '', 'nc: 2', "names: ['earth_block', 'sky_block']")
    [System.IO.File]::WriteAllLines((Join-Path $Output 'data.yaml'), $yaml, [System.Text.UTF8Encoding]::new($false))
    $report = [ordered]@{
        old_source = [System.IO.Path]::GetFullPath($OldData)
        new_source = [System.IO.Path]::GetFullPath($NewData)
        output = [System.IO.Path]::GetFullPath($Output)
        duplicate_images_removed = $duplicates
        duplicate_label_conflicts = @($labelConflicts)
        boxes_per_class = @{ earth_block = $boxes['0']; sky_block = $boxes['1'] }
        splits = $summary
    }
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'merge_report.json') -Encoding UTF8
} catch {
    Remove-Item -LiteralPath $Output -Recurse -Force
    throw
}

Write-Output "Merged dataset created: $Output"
Write-Output "Duplicate images removed: $duplicates"
Write-Output "Duplicate images with differing annotations: $($labelConflicts.Count)"
$summary | Format-Table -AutoSize
