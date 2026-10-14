param(
    [switch]$FullPipeline,
    [switch]$Promote,
    [switch]$SkipPrediction
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Results = @()

function Add-Result {
    param(
        [string]$Step,
        [string]$Status,
        [string]$Details = ""
    )

    $script:Results += [PSCustomObject]@{
        Etape   = $Step
        Statut  = $Status
        Details = $Details
    }
}

function Run-Test {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "TEST : $Name" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan

    try {
        & $Command

        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "Code de sortie : $LASTEXITCODE"
        }

        Write-Host "[OK] $Name" -ForegroundColor Green
        Add-Result $Name "PASS"
    }
    catch {
        Write-Host "[ECHEC] $Name" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red

        Add-Result $Name "FAIL" $_.Exception.Message

        return
    }
}

try {

    # =========================================================
    # 0. Vérification du dépôt
    # =========================================================

    if (-not (Test-Path ".\dvc.yaml")) {
        throw "Le script doit être exécuté depuis la racine du projet."
    }

    Write-Host ""
    Write-Host "PROJET MLOPS METEO - TEST E2E" -ForegroundColor Yellow
    Write-Host "=============================================" -ForegroundColor Yellow

    # =========================================================
    # 1. Docker
    # =========================================================

    Run-Test "Docker Engine" {

        docker version --format "{{.Server.Version}}"

    }

    # =========================================================
    # 2. Tests unitaires
    # =========================================================

    Run-Test "Tests unitaires Pytest" {

        docker run --rm `
          --env-file .env `
          -e PYTHONPATH=/workspace/src:/workspace/src/features:/workspace/src/models `
          -v "${PWD}:/workspace" `
          -w /workspace `
          models:latest `
          sh -lc "python -m pip install --no-cache-dir -q pytest && python -m pytest -q -p no:cacheprovider"

    }

    # =========================================================
    # 3. PostgreSQL
    # =========================================================

    Run-Test "PostgreSQL" {

        $status = docker inspect weather-postgres `
          --format "{{.State.Status}}"

        if ($status -ne "running") {
            throw "weather-postgres n'est pas démarré."
        }

        docker exec weather-postgres psql `
          -U weather `
          -d weather `
          -c "SELECT COUNT(*) AS total_rows FROM weather_data_raw;"

    }

    # =========================================================
    # 4. API health
    # =========================================================

    Run-Test "API Health" {

        curl.exe -k -f -sS https://localhost/health

    }

    # =========================================================
    # 5. API replicas
    # =========================================================

    Run-Test "Instances FastAPI" {

        $containers = @(
            docker ps `
              --filter "name=api-weather" `
              --format "{{.Names}}"
        )

        if ($containers.Count -eq 0) {
            throw "Aucune instance api-weather trouvée."
        }

        $healthy = 0

        foreach ($container in $containers) {

            $health = docker inspect $container `
              --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}"

            Write-Host "$container -> $health"

            if ($health -eq "healthy") {
                $healthy++
            }
        }

        if ($healthy -eq 0) {
            throw "Aucune API healthy."
        }

        Write-Host "$healthy API healthy."

    }

    # =========================================================
    # 6. DVC status
    # =========================================================

    Run-Test "DVC Status" {

        docker run --rm `
          --env-file .env `
          -e PYTHONPATH=/app/src:/app/src/features:/app/src/models `
          -v "${PWD}:/app" `
          -v "${PWD}\data:/data" `
          -v "${PWD}\models:/models" `
          -v "${PWD}\metrics:/metrics" `
          -v "${PWD}\logs:/logs" `
          -w /app `
          models:latest `
          dvc status

    }

    # =========================================================
    # 7. Pipeline ML complet OPTIONNEL
    # =========================================================

    if ($FullPipeline) {

        Run-Test "Pipeline DVC complet" {

            docker run --rm `
              --network weather `
              --env-file .env `
              -e PYTHONPATH=/app/src:/app/src/features:/app/src/models `
              -v "${PWD}:/app" `
              -v "${PWD}\data:/data" `
              -v "${PWD}\models:/models" `
              -v "${PWD}\metrics:/metrics" `
              -v "${PWD}\logs:/logs" `
              -w /app `
              models:latest `
              dvc repro

        }

        # =====================================================
        # 8. Artefacts ML
        # =====================================================

        Run-Test "Artefacts ML" {

            $artifacts = @(
                ".\data\raw\weather.parquet",
                ".\data\processed\dataset.joblib",
                ".\models\preprocessor.joblib",
                ".\models\best_params.joblib",
                ".\models\candidate_model.joblib",
                ".\metrics\scores.json",
                ".\data\predictions.csv"
            )

            foreach ($artifact in $artifacts) {

                if (-not (Test-Path $artifact)) {
                    throw "Artefact manquant : $artifact"
                }

                $file = Get-Item $artifact

                Write-Host "$($file.Name) -> $($file.Length) octets"
            }

        }

        # =====================================================
        # 9. Promotion OPTIONNELLE
        # =====================================================

        if ($Promote) {

            Run-Test "Promotion du modèle" {

                docker run --rm `
                  --network weather `
                  --env-file .env `
                  -e PYTHONPATH=/app/src:/app/src/features:/app/src/models `
                  -v "${PWD}:/app" `
                  -v "${PWD}\data:/data" `
                  -v "${PWD}\models:/models" `
                  -v "${PWD}\metrics:/metrics" `
                  -v "${PWD}\logs:/logs" `
                  -w /app `
                  models:latest `
                  python -m src.models.promote_model

            }

        }
        else {

            Add-Result "Promotion du modèle" "SKIP" "Utiliser -Promote"

        }

    }
    else {

        Add-Result "Pipeline DVC complet" "SKIP" "Utiliser -FullPipeline"

    }

    # =========================================================
    # 10. Prédiction réelle
    # =========================================================

    if (-not $SkipPrediction) {

        Run-Test "Prédiction E2E" {

            $apiUser = (
                docker exec api-weather `
                printenv API_AUTH_USERNAME
            ).Trim()

            if ([string]::IsNullOrWhiteSpace($apiUser)) {
                throw "API_AUTH_USERNAME introuvable."
            }

            $securePassword = Read-Host `
              "Mot de passe API" `
              -AsSecureString

            $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
                $securePassword
            )

            try {

                $apiPassword = `
                  [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)

                $sample = docker exec weather-postgres psql `
                  -U weather `
                  -d weather `
                  -t -A `
                  -F "|" `
                  -c "SELECT date, location, rain_tomorrow
                      FROM weather_data_raw
                      WHERE rain_tomorrow IN ('Yes','No')
                      AND date IS NOT NULL
                      AND location IS NOT NULL
                      ORDER BY date DESC
                      LIMIT 1;"

                $parts = $sample.Trim().Split("|")

                $date = $parts[0]
                $location = $parts[1]
                $actual = $parts[2]

                $encodedLocation = `
                  [uri]::EscapeDataString($location)

                $result = curl.exe -k -f -sS `
                  -u "$apiUser`:$apiPassword" `
                  "https://localhost/predict/from-db?date=$date&location=$encodedLocation"

                Write-Host $result

                $json = $result | ConvertFrom-Json

                if (-not $json.rain_tomorrow_pred) {
                    throw "Réponse de prédiction invalide."
                }

                Write-Host ""
                Write-Host "Date       : $date"
                Write-Host "Location   : $location"
                Write-Host "Réel       : $actual"
                Write-Host "Prédiction : $($json.rain_tomorrow_pred)"
                Write-Host "Probabilité: $($json.probability)"

            }
            finally {

                if ($ptr -ne [IntPtr]::Zero) {
                    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
                }

                $apiPassword = $null
                $securePassword = $null
            }

        }

    }
    else {

        Add-Result "Prédiction E2E" "SKIP"

    }

    # =========================================================
    # 11. Prometheus
    # =========================================================

    Run-Test "Prometheus" {

        $response = Invoke-WebRequest `
          "http://localhost:9090/-/ready" `
          -UseBasicParsing

        if ($response.StatusCode -ne 200) {
            throw "Prometheus non disponible."
        }

        $targets = (
            Invoke-RestMethod `
            "http://localhost:9090/api/v1/targets"
        ).data.activeTargets

        $down = @(
            $targets |
            Where-Object { $_.health -ne "up" }
        )

        $targets |
          Select-Object `
            @{N="Job";E={$_.labels.job}},
            health,
            scrapeUrl |
          Format-Table -AutoSize

        if ($down.Count -gt 0) {
            throw "$($down.Count) target(s) Prometheus DOWN."
        }

        Write-Host "$($targets.Count) targets Prometheus UP"

    }

    # =========================================================
    # 12. Métrique métier
    # =========================================================

    Run-Test "Métrique prédictions Prometheus" {

        $query = [uri]::EscapeDataString(
            'sum by (prediction) (weather_model_predictions_total)'
        )

        $result = (
            Invoke-RestMethod `
            "http://localhost:9090/api/v1/query?query=$query"
        ).data.result

        if (-not $result) {
            throw "weather_model_predictions_total absente."
        }

        $result |
          ConvertTo-Json -Depth 6 |
          Write-Host

    }

    # =========================================================
    # 13. Grafana
    # =========================================================

    Run-Test "Grafana" {

        $grafana = Invoke-RestMethod `
          "http://localhost:3000/api/health"

        $grafana | ConvertTo-Json | Write-Host

        if ($grafana.database -ne "ok") {
            throw "Grafana database != ok"
        }

    }

    # =========================================================
    # 14. Git
    # =========================================================

    Run-Test "Git" {

        Write-Host "Branche : $(git branch --show-current)"
        git status --short

    }

}
catch {

    Write-Host ""
    Write-Host "Le test E2E s'est arrêté." -ForegroundColor Red

}

# =============================================================
# Résumé
# =============================================================

Write-Host ""
Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "              RESULTAT FINAL" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Yellow

$Results | Format-Table -AutoSize

$pass = @(
    $Results |
    Where-Object Statut -eq "PASS"
).Count

$fail = @(
    $Results |
    Where-Object Statut -eq "FAIL"
).Count

$skip = @(
    $Results |
    Where-Object Statut -eq "SKIP"
).Count

Write-Host ""
Write-Host "PASS : $pass" -ForegroundColor Green
Write-Host "FAIL : $fail" -ForegroundColor Red
Write-Host "SKIP : $skip" -ForegroundColor Yellow

if ($fail -gt 0) {
    exit 1
}

exit 0
