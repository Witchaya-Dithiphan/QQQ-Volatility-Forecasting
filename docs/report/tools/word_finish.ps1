# Open a .docx in Word, refresh the three tables of contents, save, and export a PDF beside it.
param([string]$DocxPath, [string]$PdfPath)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($DocxPath, $false, $false)
    # Filling the tables changes how many pages the front matter takes, so the page
    # numbers are refreshed a second time after the document is laid out again.
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    $doc.SaveAs2($PdfPath, 17)
    "pages=" + $doc.ComputeStatistics(2) + " tables_of_contents=" + $doc.TablesOfContents.Count
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
