# pi-edge-ai

### Project Goal


### Hardware list
| Component | Specification |required|
|---|---|---|
| Raspberry Pi 4 B ||:heavy_check_mark:|                             |
| Power supply ||:heavy_check_mark: |
| Storage	microSD (A2) or USB-3 SSD| ≥ 16 GB free|:heavy_check_mark: |
| Operating system | Raspberry Pi OS 64-bit (Trixie), Desktop or Lite|:heavy_check_mark: |
| USB webcam | UVC-compatible (any standard webcam)|:heavy_check_mark: |

### Check the system (Raspberry Pi)
ssh to the pi

```bash
uname -m                      # must print aarch64, not armv7l
python3 --version             # e.g. Python 3.13.5
ldd --version | head -1       # glibc must be ≥ 2.27
free -h                       # available RAM
```
If `uname -m` shows **armv7l**, you're on a 32-bit OS. There are no 32-bit wheels for `ai-edge-litert` — that means a reinstall with the 64-bit image.

### Update the system
Update the system and install the base packages:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-opencv python3-picamera2 v4l-utils
```
### Check the camera
## 4. Step 1 — Check the camera

Plug in the USB webcam, then:

```bash
v4l2-ctl --list-devices
```

Your webcam should show up as `/dev/video0` (sometimes `/dev/video1`). Also inspect the supported formats:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

What matters is that **MJPG** is in the list. With YUYV only, a 640×480 webcam on the Pi often manages just 5–10 FPS because USB bandwidth becomes the limit.

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
