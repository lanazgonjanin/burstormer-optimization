# Creates the labels, milestones and issues described in docs/TICKETS.md on the GitHub repository.
#
# Prerequisites: GitHub CLI installed and logged in (run `gh auth login` once).
# Usage (from the repository root):
#     powershell -ExecutionPolicy Bypass -File docs\create_github_issues.ps1            # create everything
#     powershell -ExecutionPolicy Bypass -File docs\create_github_issues.ps1 -DryRun    # only print what would be made
# Issues whose title already exists on the repository are skipped, so the script is safe to re-run.

param(
    [string]$Repo = 'lanazgonjanin/burstormer-optimization',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$gh = (Get-Command gh -ErrorAction SilentlyContinue).Source
if (-not $gh) { $gh = 'C:\Program Files\GitHub CLI\gh.exe' }
if (-not (Test-Path $gh)) { throw 'GitHub CLI not found. Install it from https://cli.github.com' }

$labels = [ordered]@{
    'baseline'           = @('5319e7', 'Getting a model and evaluation we can trust')
    'track-a-ref-frame'  = @('1d76db', 'Reference-frame selection')
    'track-b-robustness' = @('d93f0b', 'Limits of the method under degraded input')
    'mitigation'         = @('0e8a16', 'Methods that move a limit')
    'training'           = @('fbca04', 'Needs a training or fine-tuning run')
    'infra'              = @('c5def5', 'Evaluation protocol, tooling, reporting')
    'stretch'            = @('bfdadc', 'Only if the earlier tickets justify it')
    'bug'                = @('d73a4a', "Something isn't working")
}
$milestones = @('M0 Trustworthy baseline', 'M1 Reference-frame selection', 'M2 Limits', 'M3 Mitigation')

# Parse docs/TICKETS.md: one ticket per '## ' heading
$tickets = @()
$current = $null
foreach ($line in Get-Content (Join-Path $PSScriptRoot 'TICKETS.md') -Encoding UTF8) {
    if ($line -match '^## (.+)$') {
        if ($current) { $tickets += $current }
        $current = @{ Title = $Matches[1].Trim(); Labels = @(); Milestone = ''; Body = New-Object System.Collections.Generic.List[string] }
    } elseif ($current) {
        if ($line -match '^Labels:\s*(.+)$') { $current.Labels = $Matches[1].Split(',') | ForEach-Object { $_.Trim() } }
        elseif ($line -match '^Milestone:\s*(.+)$') { $current.Milestone = $Matches[1].Trim() }
        else { $current.Body.Add($line) }
    }
}
if ($current) { $tickets += $current }
Write-Host "Parsed $($tickets.Count) tickets from TICKETS.md"

if ($DryRun) {
    foreach ($t in $tickets) { Write-Host "  [$($t.Milestone)] $($t.Title)  {$($t.Labels -join ', ')}" }
    return
}

foreach ($name in $labels.Keys) {
    & $gh label create $name --repo $Repo --color $labels[$name][0] --description $labels[$name][1] --force | Out-Null
}
$existingMilestones = & $gh api "repos/$Repo/milestones?state=all" --jq '.[].title'
foreach ($m in $milestones) {
    if ($existingMilestones -notcontains $m) { & $gh api "repos/$Repo/milestones" -f title=$m | Out-Null }
}
$existingTitles = & $gh issue list --repo $Repo --state all --limit 500 --json title --jq '.[].title'

foreach ($t in $tickets) {
    if ($existingTitles -contains $t.Title) { Write-Host "skip (exists): $($t.Title)"; continue }
    $bodyFile = New-TemporaryFile
    ($t.Body -join "`n").Trim() + "`n`nPlan: docs/PLAN.md" | Set-Content -Path $bodyFile -Encoding utf8
    $ghArgs = @('issue', 'create', '--repo', $Repo, '--title', $t.Title, '--body-file', $bodyFile.FullName,
                '--milestone', $t.Milestone)
    foreach ($l in $t.Labels) { $ghArgs += @('--label', $l) }
    & $gh @ghArgs
    Remove-Item $bodyFile
}
