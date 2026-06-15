[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^\d+\.\d+\.\d+([a-zA-Z0-9.-]+)?$')]
    [string]$Version,

    [switch]$SkipChecks,
    [switch]$NoCommit,
    [switch]$NoTag,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$TagName = "v$Version"
$Today = Get-Date -Format "yyyy-MM-dd"
$PyprojectPath = Join-Path $RepoRoot "pyproject.toml"
$ChangelogPath = Join-Path $RepoRoot "CHANGELOG.md"

function Invoke-CommandStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,

        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$Arguments = @()
    )

    $rendered = @($Command) + $Arguments
    Write-Host "==> $Label"
    Write-Host "    $($rendered -join ' ')"

    if ($DryRun) {
        return
    }

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $($rendered -join ' ')"
    }
}

function Write-Utf8File {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Assert-CleanTrackedWorktree {
    $status = git status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect git status."
    }

    if ($status) {
        throw "Tracked files have uncommitted changes. Commit or stash them before preparing a release."
    }
}

function Update-PyprojectVersion {
    $content = Get-Content -Raw $PyprojectPath
    $pattern = '(?m)^version = "([^"]+)"$'

    if ($content -notmatch $pattern) {
        throw "Could not find project version in pyproject.toml."
    }

    $currentVersion = $Matches[1]
    if ($currentVersion -eq $Version) {
        throw "pyproject.toml is already at version $Version."
    }

    $regex = [regex]::new($pattern)
    $updated = $regex.Replace($content, "version = `"$Version`"", 1)

    Write-Host "==> Update pyproject.toml version"
    Write-Host "    $currentVersion -> $Version"

    if (-not $DryRun) {
        Write-Utf8File $PyprojectPath $updated
    }
}

function Update-Changelog {
    if (-not (Test-Path $ChangelogPath)) {
        throw "CHANGELOG.md does not exist."
    }

    $content = Get-Content -Raw $ChangelogPath
    if ($content -match "(?m)^## \[$([regex]::Escape($Version))\]") {
        throw "CHANGELOG.md already contains a section for $Version."
    }

    $pattern = '(?s)## \[Unreleased\]\r?\n(?<body>.*?)(?=\r?\n## \[|\z)'
    $match = [regex]::Match($content, $pattern)
    if (-not $match.Success) {
        throw "Could not find an Unreleased section in CHANGELOG.md."
    }

    $body = $match.Groups["body"].Value.Trim()
    if ([string]::IsNullOrWhiteSpace($body) -or $body -eq "- Nothing yet.") {
        $body = "- No changes documented."
    }

    $replacement = "## [Unreleased]`r`n`r`n- Nothing yet.`r`n`r`n## [$Version] - $Today`r`n`r`n$body`r`n"
    $regex = [regex]::new($pattern)
    $updated = $regex.Replace($content, $replacement, 1)

    Write-Host "==> Update CHANGELOG.md"
    Write-Host "    Move Unreleased notes to $Version"

    if (-not $DryRun) {
        Write-Utf8File $ChangelogPath $updated
    }
}

function Assert-TagDoesNotExist {
    git rev-parse -q --verify "refs/tags/$TagName" *> $null
    if ($LASTEXITCODE -eq 0) {
        throw "Tag $TagName already exists."
    }
}

function Test-BuildArtifacts {
    $artifactPaths = Get-ChildItem -Path (Join-Path $RepoRoot "dist") -File -Filter "visionpack-$Version*" |
        ForEach-Object { $_.FullName }

    if (-not $artifactPaths) {
        throw "No dist artifacts found for version $Version."
    }

    Invoke-CommandStep "Validate distribution metadata" "uvx" (@("twine", "check") + $artifactPaths)
}

if ($NoCommit -and -not $NoTag) {
    throw "Use -NoTag when using -NoCommit; otherwise the tag would not include the release changes."
}

if (-not $DryRun) {
    Assert-CleanTrackedWorktree
}
Assert-TagDoesNotExist
Update-PyprojectVersion
Update-Changelog

if (-not $SkipChecks) {
    Invoke-CommandStep "Run Ruff" "uv" @("run", "ruff", "check", ".")
    Invoke-CommandStep "Run unit tests" "uv" @("run", "python", "-m", "unittest", "discover", "-s", "tests", "-q")
}

Invoke-CommandStep "Build source distribution and wheel" "uv" @("build")

if (-not $DryRun) {
    Test-BuildArtifacts
}

if (-not $NoCommit) {
    Invoke-CommandStep "Stage release files" "git" @("add", "pyproject.toml", "CHANGELOG.md")
    Invoke-CommandStep "Create release commit" "git" @("commit", "-m", "Release $TagName")
}

if (-not $NoTag) {
    Invoke-CommandStep "Create release tag" "git" @("tag", $TagName)
}

Write-Host ""
Write-Host "Release $TagName is prepared."
Write-Host "Next steps:"
Write-Host "  git push origin HEAD"
Write-Host "  git push origin $TagName"
Write-Host "  gh release create $TagName --draft --title `"$TagName`" --notes-file CHANGELOG.md"
Write-Host "  Publish the GitHub Release when ready; the publish workflow will upload to PyPI."
