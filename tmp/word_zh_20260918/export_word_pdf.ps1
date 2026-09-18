param([Parameter(Mandatory=$true)][string]$InputDocument,
      [Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$documentPath = [System.IO.Path]::GetFullPath($InputDocument)
$pdfPath = [System.IO.Path]::GetFullPath($OutputPdf)
$parentDirectory = [System.IO.Path]::GetDirectoryName($pdfPath)
[System.IO.Directory]::CreateDirectory($parentDirectory) | Out-Null
$wordTaskApp = $null
$wordTaskDocument = $null
try {
    $wordTaskApp = New-Object -ComObject Word.Application
    $wordTaskApp.Visible = $false
    $wordTaskApp.DisplayAlerts = 0
    $wordTaskDocument = $wordTaskApp.Documents.Open($documentPath, $false, $true, $false)
    $wordTaskDocument.Repaginate()
    $pages = $wordTaskDocument.ComputeStatistics(2)
    $wordTaskDocument.ExportAsFixedFormat($pdfPath, 17)
    [pscustomobject]@{Pages=$pages;Pdf=$pdfPath} | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $wordTaskDocument) {
        $wordTaskDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordTaskDocument)
    }
    if ($null -ne $wordTaskApp) {
        $wordTaskApp.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordTaskApp)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
