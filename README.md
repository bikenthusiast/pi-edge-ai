# pi-edge-ai

Edge-AI-Pipeline auf einem Raspberry Pi 4 B mit Bedrock-gestütztem Agenten.

Ein Repository, vier Zonen: entwickelt wird auf dem Mac, deployt wird eine
Teilmenge auf den Pi, das Reasoning läuft in AWS Bedrock.

| Zone | Inhalt | Wie es dorthin kommt |
|---|---|---|
| Mac | alles | `git clone` |
| Git | alles außer Gewichten und Secrets | `git push` |
| Pi | `src/`, `models/`, `requirements-pi.txt` | `make deploy` (rsync) |
| AWS | keine Dateien, nur API-Aufrufe | `infra/`, vom Mac aus |

## Schnellstart (Mac)

```bash
make setup      # venv + Dev-Abhängigkeiten
make models     # Gewichte laden (nicht im Repo)
make test       # Testsuite ohne Hardware
```

## Erstes Deployment (Pi)

```bash
make deploy                                   # rsync src/ + models/
ssh pi 'cd pi-edge-ai && bash scripts/bootstrap_pi.sh'
make run-pi                                   # Gegenprobe über SSH
```

## Struktur

```
src/edge/vision/   Modell laden, Preprocessing, Kamera
src/edge/events/   SQLite-Eventlog
src/edge/agent/    Strands-Tools → Bedrock
tests/             läuft auf dem Mac, wird nie deployt
infra/             IAM, Knowledge Base
docs/adr/          Entscheidungsprotokolle
```

## Entscheidungen

- [ADR 0001](docs/adr/0001-litert-statt-tflite-runtime.md) — LiteRT statt `tflite-runtime`
- [ADR 0002](docs/adr/0002-agent-laeuft-auf-dem-pi.md) — Agent auf dem Pi statt in Lambda
