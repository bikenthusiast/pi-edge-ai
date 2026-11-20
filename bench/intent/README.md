# Intent-Benchmark: Sprachbefehle mit kleinen LLMs auf dem Pi 4

Wie zuverlässig und wie schnell macht ein lokales LLM auf dem Raspberry Pi 4 aus einem
frei gesprochenen Satz einen ausführbaren Befehl? Und was bringt Architektur
gegenüber einem größeren Modell oder neuer Hardware?

Ergebnis und Entscheidung: [ADR 0005](../../docs/adr/0005-sprachbefehle-regeln-llm-pruefschicht.md).

## Dateien

| Datei | Zweck |
|---|---|
| `llm_bench.py` | Benchmark: Modelle × Prompt-Varianten × Testsätze, Ollama oder llama-server; `compare` stellt Läufe nebeneinander |
| `fastpath.py` | Regeln für eindeutige Befehle, `parse_minutes()` für deutsche Zeitangaben, Prüfschicht `guard()`; Selbsttest mit `python3 fastpath.py` |
| `holdout.csv` | 26 Testsätze, die beim Schreiben von Prompt v2 und den Regeln nicht bekannt waren (inzwischen zum Nachbessern verwendet) |
| `record_testset.py` | Nimmt einen frischen Testsatz auf: Aufgabenkarten statt vorgegebener Sätze, freie Vosk-Transkription, CSV im selben Format |
| `results/` | Rohdaten der Läufe (JSON) und `pi4.md`, die Vergleichstabelle aus `compare` |

Die Skripte laufen nicht über `make deploy`, sondern werden direkt auf den Pi
kopiert (`scp bench/intent/*.py bench/intent/*.csv pi:~/`). Sie brauchen nur `httpx`
und `pydantic`, die `uv` pro Aufruf holt.

## Ablauf

```bash
# Ollama (Standard)
uv run --with httpx --with pydantic llm_bench.py run --label pi4 \
    --prompts v2g --models qwen2.5:1.5b --fastpath --cases holdout.csv

# llama-server mit derselben GGUF-Datei
uv run --with httpx --with pydantic llm_bench.py run --label pi4-llamacpp --backend llamacpp \
    --prompts v2g --models qwen2.5:1.5b --fastpath --cases holdout.csv

# Vergleich (erste Datei = Basis für den Beschleunigungsfaktor)
uv run --with httpx --with pydantic llm_bench.py compare results/*.json --md results/pi4.md
```

Optionen: `--runs` (Wiederholungen; bei `temperature 0` nur für stabilere Zeiten),
`--cooldown` (Pause zwischen Modellen), `selftest` (Regeln und Prüfschicht ohne LLM).

### Prompt-Varianten

| Variante | Inhalt |
|---|---|
| `v1` | Aktionen beschreiben, Modell rechnet Minuten selbst |
| `v2` | + zehn Beispiele, klare Regeln, Zeitangabe wörtlich (`duration`), Minuten im Code |
| `v2g` | v2 + Prüfschicht `guard()` gegen den Originalsatz |
| `v3` | v2 + `timer_stop`, `spotify_now_playing` im Prompt + Prüfschicht |

`--fastpath` setzt die Regeln vor das LLM (in Tabellen als `+fp`).

### Kennzahlen

- **Fehlaktionen:** Jarvis hätte etwas Falsches *getan* — falsche Aktion mit Wirkung
  (Musik, Licht, Timer) oder richtige Aktion mit falschem Wert (52 statt 25 Minuten,
  falsche Playlist). `unknown` oder eine Titelansage zählen nicht. Die wichtigste Zahl.
- **Treffer:** Aktion und alle Werte richtig. **Aktion:** nur die Aktion richtig.
- **Regeln:** Anteil der Sätze, die ohne LLM erkannt wurden. **Prüfung:** Eingriffe von `guard()`.
- **Median / p90:** Zeit pro Satz inklusive Regel-Treffern. **Kaltstart:** erster Aufruf
  inklusive Laden des Modells und Verarbeiten des System-Prompts.
- **Tmax / Drosselung:** höchste SoC-Temperatur und `vcgencmd get_throttled` während des Laufs.

## Ergebnisse (Raspberry Pi 4B, 8 GB, 18.11.2026)

Ollama 0.40.16, Q4_K_M, 4 Threads, `temperature 0`. Der Spiegel lief während der
Messungen nicht (eigene SD-Karte).

### Modellgröße (Prompt v1, 15 Sätze × 3)

| Modell | Fehlaktionen | Treffer | Median | p90 | tok/s | Kaltstart | RAM | Tmax |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 18 von 45 | 60 % | 2,4 s | 3,5 s | 8,1 | 22 s | 0,45 GB | 53 °C |
| qwen2.5:1.5b | 6 von 45 | 87 % | 4,9 s | 8,0 s | 3,4 | 47 s | 1,1 GB | 57 °C |
| qwen2.5:3b | 6 von 45 | 87 % | 9,1 s | 15,3 s | 1,8 | 117 s | 2,0 GB | 58 °C |

### Prompt und Regeln (qwen2.5:1.5b)

| Testsatz | Variante | Regeln | Fehlaktionen | Treffer | Median | p90 |
|---|---|---|---|---|---|---|
| basis (15) | v1 | – | 2 | 87 % | 4,76 s | 7,91 s |
| basis (15) | v2 | – | 0 | 100 % | 3,75 s | 7,89 s |
| basis (15) | v2 + Regeln | 67 % | 0 | 100 % | < 0,01 s | 6,35 s |
| holdout (26) | v2 + Regeln, erster Lauf | 42 % | 4 | 85 % | 3,28 s | 6,85 s |
| holdout (26) | v2 + Regeln¹ | 50 % | 2 | 92 % | 1,61 s | 6,87 s |
| holdout (26) | **v2g + Regeln¹** | 50 % | **0** | **100 %** | 1,59 s | 6,86 s |
| holdout (26) | v3 + Regeln¹ | 50 % | 0 | 92 % | 1,61 s | 7,14 s |

¹ nach dem Nachbessern auf Basis des ersten Holdout-Laufs (Regeln für `timer_stop`
und `spotify_now_playing`, Prüfschicht), also nicht mehr unabhängig.

### Laufzeit: Ollama gegen llama.cpp (gleiche GGUF-Datei, v2g + Regeln, holdout)

| Laufzeit | Fehlaktionen | Treffer | Median | p90 | tok/s |
|---|---|---|---|---|---|
| Ollama 0.40.16 | 0 | 100 % | 1,59 s | 6,86 s | 3,6 |
| llama-server (b1-6f311d7) | 0 | 100 % | 1,59 s | 6,88 s | 3,6 |

`llama-bench -t 4 -p 256 -n 32`: 9,02 tok/s beim Prompt, 3,39 tok/s bei der Erzeugung.
Der Pi 4 ist durch Speicherbandbreite und fehlende Dot-Product-Befehle des Cortex-A72
begrenzt, nicht durch die Laufzeitumgebung.

## Was die Fehler zeigten

| Fehler | Ursache | Lösung |
|---|---|---|
| „fünfundzwanzig“ → 52 | Deutsche Zahlwörter nennen die Einer zuerst | Modell gibt Zeitangabe wörtlich zurück, `parse_minutes()` rechnet |
| „Musik lauter“ → Playlist „Laut“ | Keine passende Aktion, Modell nimmt die nächstbeste | Beispiele mit `unknown` im Prompt, Prüfschicht: Playlist muss im Satz vorkommen |
| „play liste feier abend“ | Spracherkennung trennt Wörter | Bekannte Playlists ohne Leerzeichen abgleichen |
| „Chill am Abend“ → „Chill“ | Mehrteiliger Name abgeschnitten | Namen nach dem Wort „Playlist“ aus dem Satz ergänzen |
| „Was läuft gerade?“ → Musik starten | Frage als Befehl gelesen | Fragen lösen keine Aktion aus |
| „Stopp den Timer“ → Timer 1 Minute | Aktion fehlte, Wert erfunden | Regel für `timer_stop`; ohne Zeitangabe im Satz kein Timer |
| v3: „Mach es hell“ → `unknown` | Mehr Aktionen und Regeln im Prompt verunsichern das 1,5B-Modell | Neue Aktionen in die Regeln, Prompt bei v2 lassen |

## Methodik und Grenzen

- **Testsätze:** `basis` (15, im Skript) diente zur Entwicklung von Prompt v2 und den
  Regeln. `holdout.csv` (26) war dabei unbekannt; der erste Holdout-Lauf ist die
  ehrliche Messung (85 % Treffer, 4 Fehlaktionen). Danach wurde auf dieser Basis
  nachgebessert — die 100 % gelten deshalb nur für gesehene Sätze.
- **Nächster Schritt:** frischer Testsatz mit `record_testset.py`. Die Sätze
  formuliert der Sprecher frei zu Aufgabenkarten, Vosk transkribiert ohne Grammatik,
  und niemand verändert danach Prompt, Regeln oder Prüfschicht, bevor gemessen ist.
- **Ein Sprecher, eine Umgebung:** Alle Sätze stammen von einer Person; der erste
  Vosk-Satz wird im Flur vor dem Spiegel aufgenommen.
- **Hardware-Vergleich:** Werte für den Pi 5 stammen bisher aus veröffentlichten
  Messungen (qwen2.5:1.5b ~10 tok/s); ein eigener Lauf mit `--label pi5` steht aus.
