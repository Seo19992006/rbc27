param(
    [string]$Source = (Join-Path $PSScriptRoot 'xylyanh.v2i.yolov8'),
    [string]$Output = (Join-Path $PSScriptRoot 'dataset_clean')
)

$ErrorActionPreference = 'Stop'
$classes = @('earth_block', 'sky_block')
$splits = @('train', 'valid', 'test')

if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
    throw "Source folder not found: $Source"
}
foreach ($split in $splits) {
    foreach ($folder in @('images', 'labels')) {
        $path = Join-Path (Join-Path $Source $split) $folder
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "Missing dataset folder: $path"
        }
    }
}
if (Test-Path -LiteralPath $Output) {
    throw "Output folder already exists; choose another -Output to avoid overwriting: $Output"
}

$counts = @{ '0' = 0; '1' = 0 }
$summary = [System.Collections.Generic.List[object]]::new()
New-Item -ItemType Directory -Path $Output | Out-Null
try {
    foreach ($split in $splits) {
        $sourceImages = Join-Path (Join-Path $Source $split) 'images'
        $sourceLabels = Join-Path (Join-Path $Source $split) 'labels'
        $targetImages = Join-Path (Join-Path $Output $split) 'images'
        $targetLabels = Join-Path (Join-Path $Output $split) 'labels'
        New-Item -ItemType Directory -Path $targetImages, $targetLabels | Out-Null

        $images = @(Get-ChildItem -LiteralPath $sourceImages -File | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' })
        $labelFiles = @(Get-ChildItem -LiteralPath $sourceLabels -File -Filter '*.txt')
        $imageStems = @{}
        foreach ($image in $images) {
            if ($imageStems.ContainsKey($image.BaseName)) {
                throw "Duplicate image stem: $split/$($image.BaseName)"
            }
            $imageStems[$image.BaseName] = $true
        }
        foreach ($image in $images) {
            $labelPath = Join-Path $sourceLabels ($image.BaseName + '.txt')
            if (-not (Test-Path -LiteralPath $labelPath -PathType Leaf)) {
                throw "Image has no label: $split/$($image.Name)"
            }
            Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $targetImages $image.Name)
            $cleanLines = [System.Collections.Generic.List[string]]::new()
            foreach ($line in Get-Content -LiteralPath $labelPath) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                $parts = $line.Trim() -split '\s+'
                if ($parts.Count -ne 5) { throw "Label must have 5 columns at $split/$($image.BaseName).txt: $line" }
                $classId = 0
                if (-not [int]::TryParse($parts[0], [ref]$classId) -or $classId -notin 0, 1) {
                    throw "Invalid class ID at $split/$($image.BaseName).txt: $line"
                }
                $values = [System.Collections.Generic.List[double]]::new()
                foreach ($valueText in $parts[1..4]) {
                    $value = 0.0
                    if (-not [double]::TryParse($valueText, [Globalization.NumberStyles]::Float, [cultureinfo]::InvariantCulture, [ref]$value) -or [double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0 -or $value -gt 1) {
                        throw "YOLO coordinate outside [0,1] at $split/$($image.BaseName).txt: $line"
                    }
                    $values.Add($value)
                }
                if ($values[2] -le 0 -or $values[3] -le 0) { throw "Box width/height is zero at $split/$($image.BaseName).txt: $line" }
                $counts["$classId"]++
                $cleanLines.Add(($classId.ToString() + ' ' + ($parts[1..4] -join ' ')))
            }
            if ($cleanLines.Count -eq 0) { throw "Empty label at $split/$($image.BaseName).txt" }
            [System.IO.File]::WriteAllLines((Join-Path $targetLabels ($image.BaseName + '.txt')), $cleanLines, [System.Text.UTF8Encoding]::new($false))
        }
        foreach ($label in $labelFiles) {
            if (-not $imageStems.ContainsKey($label.BaseName)) { throw "Label has no matching image: $split/$($label.Name)" }
        }
        $summary.Add([pscustomobject]@{ split = $split; image_count = $images.Count; label_count = $labelFiles.Count })
    }

    $yaml = @(
        'path: .'
        'train: train/images'
        'val: valid/images'
        'test: test/images'
        ''
        'nc: 2'
        "names: ['$($classes[0])', '$($classes[1])']"
    )
    [System.IO.File]::WriteAllLines((Join-Path $Output 'data.yaml'), $yaml, [System.Text.UTF8Encoding]::new($false))
    $report = [ordered]@{
        source = [System.IO.Path]::GetFullPath($Source)
        output = [System.IO.Path]::GetFullPath($Output)
        preprocessing = 'Keep image pixels; normalize YOLO label whitespace; preserve splits and box coordinates.'
        classes = $classes
        splits = $summary
        boxes_per_class = @{ earth_block = $counts['0']; sky_block = $counts['1'] }
    }
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'preprocess_report.json') -Encoding UTF8
} catch {
    Remove-Item -LiteralPath $Output -Recurse -Force
    throw
}

Write-Output "Clean dataset created: $Output"
$summary | Format-Table -AutoSize
Write-Output "Boxes per class: earth_block=$($counts['0']), sky_block=$($counts['1'])"
