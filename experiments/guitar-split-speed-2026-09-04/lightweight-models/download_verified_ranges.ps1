param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [Parameter(Mandatory = $true)]
    [long]$Size,

    [Parameter(Mandatory = $true)]
    [string]$Sha256,

    [int]$Parts = 8
)

$ErrorActionPreference = 'Stop'
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDir = [System.IO.Path]::GetDirectoryName($outputPath)
$partsDir = "$outputPath.parts"
$stagingPath = "$outputPath.staging"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $partsDir | Out-Null

$chunkSize = [long][Math]::Ceiling($Size / [double]$Parts)
$curlArgs = @(
    '--parallel',
    '--parallel-immediate',
    '--parallel-max', $Parts.ToString(),
    '--silent',
    '--show-error'
)
$transferCount = 0

for ($index = 0; $index -lt $Parts; $index++) {
    $start = [long]$index * $chunkSize
    $end = [Math]::Min($Size - 1, $start + $chunkSize - 1)
    $partPath = Join-Path $partsDir ('part-{0:D2}' -f $index)
    $expectedPartSize = $end - $start + 1

    if ((Test-Path -LiteralPath $partPath) -and
        (Get-Item -LiteralPath $partPath).Length -eq $expectedPartSize) {
        continue
    }

    if ($transferCount -gt 0) {
        $curlArgs += '--next'
    }

    $curlArgs += @(
        '--ssl-no-revoke',
        '--location',
        '--fail',
        '--retry', '8',
        '--retry-all-errors',
        '--connect-timeout', '20',
        '--max-time', '300',
        '--range', "$start-$end",
        '--output', $partPath,
        $Url
    )
    $transferCount++
}

if ($transferCount -gt 0) {
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed with exit code $LASTEXITCODE"
    }
}

$destination = [System.IO.File]::Create($stagingPath)
try {
    for ($index = 0; $index -lt $Parts; $index++) {
        $partPath = Join-Path $partsDir ('part-{0:D2}' -f $index)
        $source = [System.IO.File]::OpenRead($partPath)
        try {
            $source.CopyTo($destination)
        }
        finally {
            $source.Dispose()
        }
    }
}
finally {
    $destination.Dispose()
}

$actualSize = (Get-Item -LiteralPath $stagingPath).Length
if ($actualSize -ne $Size) {
    throw "Size mismatch: expected $Size, got $actualSize"
}

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $stagingPath).Hash.ToLowerInvariant()
if ($actualHash -ne $Sha256.ToLowerInvariant()) {
    throw "SHA-256 mismatch: expected $Sha256, got $actualHash"
}

Move-Item -LiteralPath $stagingPath -Destination $outputPath -Force
Get-ChildItem -LiteralPath $partsDir -File | Remove-Item -Force

[pscustomobject]@{
    Path = $outputPath
    Size = $actualSize
    Sha256 = $actualHash
}
