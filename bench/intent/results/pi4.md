# LLM-Intent-Benchmark

## Hardware

- **pi4**: Raspberry Pi 4 Model B Rev 1.4, 7.6 GB, Ollama 0.40.16, 2026-11-18T14:27:06
- **pi4**: Raspberry Pi 4 Model B Rev 1.4, 7.6 GB, Ollama 0.40.16, 2026-11-18T15:23:53
- **pi4**: Raspberry Pi 4 Model B Rev 1.4, 7.6 GB, Ollama 0.40.16, 2026-11-18T15:50:49
- **pi4**: Raspberry Pi 4 Model B Rev 1.4, 7.6 GB, Ollama 0.40.16, 2026-11-18T16:12:24
- **pi4**: Raspberry Pi 4 Model B Rev 1.4, 7.6 GB, Ollama 0.40.16, 2026-11-18T16:22:41
- **pi4**: Raspberry Pi 4 Model B Rev 1.4, 7.6 GB, Ollama 0.40.16, 2026-11-18T17:23:07

## Ergebnisse

| Hardware | Modell | Testsatz | Prompt | Regeln | Prüfung | Fehlaktionen | Treffer | Aktion | Median s | p90 s | tok/s | Kaltstart s | RAM GB | Tmax °C | Drosselung |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pi4 | qwen2.5:0.5b | basis | v1 | 0% | 0 | 18 | 60% | 73% | 2.38 | 3.50 | 8.1 | 22.2 | 0.45 | 53.0 | 0x0 |
| pi4 | qwen2.5:1.5b | basis | v1 | 0% | 0 | 6 | 87% | 93% | 4.86 | 7.97 | 3.4 | 47.1 | 1.09 | 56.9 | 0x0 |
| pi4 | qwen2.5:3b | basis | v1 | 0% | 0 | 6 | 87% | 100% | 9.13 | 15.28 | 1.8 | 116.5 | 2.02 | 58.4 | 0x0 |
| pi4 | qwen2.5:1.5b | basis | v1 | 0% | 0 | 2 | 87% | 93% | 4.76 | 7.91 | 3.5 | 25.7 | 1.09 | 56.0 | 0x0 |
| pi4 | qwen2.5:1.5b | basis | v2 | 0% | 0 | 0 | 100% | 100% | 3.75 | 7.89 | 3.6 | 45.0 | 1.09 | 58.4 | 0x0 |
| pi4 | qwen2.5:3b | basis | v1 | 0% | 0 | 2 | 87% | 100% | 9.05 | 15.35 | 1.8 | 52.0 | 2.02 | 58.9 | 0x0 |
| pi4 | qwen2.5:3b | basis | v2 | 0% | 0 | 0 | 100% | 100% | 7.48 | 15.87 | 1.8 | 91.6 | 2.02 | 60.3 | 0x0 |
| pi4 | qwen2.5:1.5b | basis | v2+fp | 67% | 0 | 0 | 100% | 100% | 0.00 | 6.35 | 3.6 | 57.8 | 1.09 | 54.5 | 0x0 |
| pi4 | qwen2.5:1.5b | holdout | v2+fp | 42% | 0 | 4 | 85% | 92% | 3.28 | 6.85 | 3.6 | 57.9 | 1.09 | 56.0 | 0x0 |
| pi4 | qwen2.5:1.5b | holdout | v2+fp | 50% | 0 | 2 | 92% | 100% | 1.58 | 6.83 | 3.6 | 57.6 | 1.09 | 56.0 | 0x0 |
| pi4 | qwen2.5:1.5b | holdout | v3+fp | 50% | 2 | 0 | 92% | 92% | 1.59 | 7.11 | 3.3 | 55.0 | 1.09 | 58.9 | 0x0 |
| pi4 | qwen2.5:1.5b | holdout | v2+fp | 50% | 0 | 2 | 92% | 100% | 1.61 | 6.87 | 3.6 | 57.4 | 1.09 | 56.4 | 0x0 |
| pi4 | qwen2.5:1.5b | holdout | v2g+fp | 50% | 2 | 0 | 100% | 100% | 1.59 | 6.86 | 3.6 | 3.3 | 1.09 | 57.9 | 0x0 |
| pi4 | qwen2.5:1.5b | holdout | v3+fp | 50% | 2 | 0 | 92% | 92% | 1.61 | 7.14 | 3.3 | 55.2 | 1.09 | 59.4 | 0x0 |

Beschleunigung gegenüber **pi4** (Median-Latenz, >1 = schneller):

| Hardware | Modell | Prompt | Faktor Latenz | Faktor tok/s |
|---|---|---|---|---|
| pi4 | qwen2.5:1.5b | v1 | 1.02× | 1.02× |
| pi4 | qwen2.5:3b | v1 | 1.01× | 1.01× |

Fehlerhafte Fälle (⚠ = Fehlaktion, hätte etwas Falsches ausgelöst):

- ⚠ pi4 / qwen2.5:0.5b / v1 / basis: 'Mach die Musik aus' → {"action": "spotify_play"} (erwartet {"action": "spotify_pause", "playlist": null, "minutes": null}) ×3
- ⚠ pi4 / qwen2.5:0.5b / v1 / basis: 'Lampe aus bitte' → {"action": "lamp_on"} (erwartet {"action": "lamp_off", "playlist": null, "minutes": null}) ×3
- ⚠ pi4 / qwen2.5:0.5b / v1 / basis: 'Timer für eine Viertelstunde' → {"action": "timer_start", "minutes": 30} (erwartet {"action": "timer_start", "playlist": null, "minutes": 15}) ×3
- ⚠ pi4 / qwen2.5:0.5b / v1 / basis: 'stell mir nen timer auf fünfundzwanzig minuten' → {"action": "timer_start", "minutes": 52} (erwartet {"action": "timer_start", "playlist": null, "minutes": 25}) ×3
- ⚠ pi4 / qwen2.5:0.5b / v1 / basis: 'Mach die Musik lauter' → {"action": "spotify_play", "playlist": "Laufplaylist", "minutes": 15} (erwartet {"action": "unknown", "playlist": null, "minutes": null}) ×3
- ⚠ pi4 / qwen2.5:0.5b / v1 / basis: 'Wie wird das Wetter morgen' → {"action": "spotify_play", "playlist": "morgen"} (erwartet {"action": "unknown", "playlist": null, "minutes": null}) ×3
- ⚠ pi4 / qwen2.5:1.5b / v1 / basis: 'stell mir nen timer auf fünfundzwanzig minuten' → {"action": "timer_start", "minutes": 52} (erwartet {"action": "timer_start", "playlist": null, "minutes": 25}) ×4
- ⚠ pi4 / qwen2.5:1.5b / v1 / basis: 'Mach die Musik lauter' → {"action": "spotify_play", "playlist": "Laut"} (erwartet {"action": "unknown", "playlist": null, "minutes": null}) ×4
- ⚠ pi4 / qwen2.5:3b / v1 / basis: 'Mach mal Musik von der Playlist Sonntagmorgen an' → {"action": "spotify_play"} (erwartet {"action": "spotify_play", "playlist": "Sonntagmorgen", "minutes": null}) ×4
- ⚠ pi4 / qwen2.5:3b / v1 / basis: 'Timer für eine Viertelstunde' → {"action": "timer_start", "minutes": 20} (erwartet {"action": "timer_start", "playlist": null, "minutes": 15}) ×4
- ⚠ pi4 / qwen2.5:1.5b / v2+fp / holdout: 'spiel die play liste feier abend' → {"action": "spotify_play", "playlist": "Play Liste Feier Abend"} (erwartet {"action": "spotify_play", "playlist": "Feierabend", "minutes": null}) ×3
- ⚠ pi4 / qwen2.5:1.5b / v2+fp / holdout: 'Spiel meine Playlist Chill am Abend' → {"action": "spotify_play", "playlist": "Chill"} (erwartet {"action": "spotify_play", "playlist": "Chill am Abend", "minutes": null}) ×3
- ⚠ pi4 / qwen2.5:1.5b / v2+fp / holdout: 'Was läuft gerade auf Spotify' → {"action": "spotify_play"} (erwartet {"action": "unknown", "playlist": null, "minutes": null}) ×1
- ⚠ pi4 / qwen2.5:1.5b / v2+fp / holdout: 'Stopp den Timer' → {"action": "timer_start", "minutes": 1} (erwartet {"action": "unknown", "playlist": null, "minutes": null}) ×1
- pi4 / qwen2.5:1.5b / v3+fp / holdout: 'Mach es hell' → {"action": "unknown"} (erwartet {"action": "lamp_on", "playlist": null, "minutes": null}) ×2
- pi4 / qwen2.5:1.5b / v3+fp / holdout: 'Mach die Musik leiser' → {"action": "spotify_now_playing"} (erwartet {"action": "unknown", "playlist": null, "minutes": null}) ×2
