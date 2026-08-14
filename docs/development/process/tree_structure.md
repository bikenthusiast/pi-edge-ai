tree structure

pi-edge-ai/
├── README.md                      # Einstieg, Architekturbild, Ergebnisse
├── Makefile                       # setup · test · deploy · run-pi
├── pyproject.toml                 # Paketdefinition, pytest, ruff
├── requirements-pi.txt            # nur was aarch64-Wheels hat
├── requirements-dev.txt           # pytest, ruff — nie deployt
├── .gitignore
│
├── docs/
│   └── adr/                       # Entscheidungsprotokolle
│       ├── 0001-litert-statt-tflite-runtime.md
│       └── 0002-agent-laeuft-auf-dem-pi.md
│
├── scripts/
│   ├── deploy.sh                  # rsync-Filter = die Zonengrenze
│   ├── bootstrap_pi.sh            # einmalig auf dem Pi
│   └── fetch_models.sh            # läuft auf beiden Maschinen
│
├── models/                        # Gewichte: gitignored, per Manifest geholt
│   └── manifest.txt
│
├── src/edge/                      # ↓ alles hierunter läuft auf dem Pi
│   ├── vision/
│   │   ├── model.py               # hardwarefrei → auf dem Mac testbar
│   │   ├── classify.py            # CLI
│   │   └── camera.py              # braucht Hardware
│   ├── events/
│   │   └── store.py               # SQLite-Eventlog
│   └── agent/
│       └── tools.py               # Strands-Tools → rufen Bedrock
│
├── tests/                         # läuft auf dem Mac, nie deployt
│   ├── test_vision.py
│   └── fixtures/
│
└── infra/                         # IAM, Knowledge Base — Mac → AWS
    ├── iam/
    └── knowledge_base/
