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
    # A Greek letter in a heading or caption keeps its Cambria font in the tables of contents but
    # takes the 16 pt entry size there, so it is set back to the 12 pt used in body text.
    foreach ($toc in $doc.TablesOfContents) {
        foreach ($name in @("Cambria", "Segoe UI Symbol")) {
            $find = $toc.Range.Find
            $find.ClearFormatting()
            $find.Replacement.ClearFormatting()
            $find.Font.Name = $name
            $find.Replacement.Font.Size = 12
            # text, case, whole word, wildcards, sounds like, word forms, forward, wrap = stop, format, replacement, replace all
            $find.Execute("", $false, $false, $false, $false, $false, $true, 0, $true, "", 2) | Out-Null
        }
    }
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
