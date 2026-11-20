# ADR 0005 — Frei formulierte Sprachbefehle: Regeln, kleines LLM mit JSON-Schema, Prüfschicht

**Status:** vorgeschlagen · **Datum:** 2026-11

## Kontext
Jarvis, der Sprachassistent im MagicMirror (`voice/` im Spiegel-Projekt), erkennt
Befehle heute mit Vosk und einer festen Grammatik (`PHRASES` → `to_intent`). Das ist
schnell, versteht aber nur die vorgesehenen Formulierungen und verwechselt ähnlich
klingende Befehle. Was die Grammatik nicht erkennt, wird frei transkribiert und an
ein lokales LLM für freie Fragen gegeben (`ask.py`, `llm.service`: llama-server mit
Gemma 3 1B, Entscheidung vom 16.11.2026). Dieses LLM **antwortet** nur, es führt
keine Aktion aus.

Ziel: Auch frei formulierte Befehle („kannst du das Licht ausmachen“, „spiel die
Playlist Chill am Abend“) sollen ausgeführt werden — lokal auf dem Pi 4, und vor
allem ohne **Fehlaktionen**: Ein Sprachassistent, der bei einer Frage Musik startet
oder einen Timer auf 52 statt 25 Minuten stellt, ist schlechter als einer, der
nachfragt.

Gemessen mit `bench/intent/llm_bench.py` am 18.11.2026 auf dem Pi 4B (8 GB):
drei Modellgrößen, drei Prompt-Varianten, mit und ohne Regeln, zwei Testsätze
(Details und alle Zahlen: [bench/intent/README.md](../../bench/intent/README.md)).

## Entscheidung
Für Sätze, die die Grammatik nicht erkennt, eine Kette aus vier Stufen vor dem
bisherigen Weg zu `ask.py`:

1. **Freie Transkription** (vorhanden, `transcribe_free`).
2. **Regeln** (`fastpath.match`): Ein Satz wird nur erkannt, wenn nach dem
   Entfernen von Füllwörtern *jedes* Wort zum Wortschatz genau einer Aktion gehört.
   Verneinungen, unbekannte Wörter, zwei Befehle in einem Satz → weiter zu Stufe 3.
3. **LLM als Intent-Parser:** Qwen 2.5 1.5B Instruct (Q4_K_M) mit Prompt v2:
   sechs Kernaktionen, Beispiele im System-Prompt, Ausgabe per JSON-Schema
   erzwungen, Zeitangaben **wörtlich** statt als Zahl.
4. **Prüfschicht** (`fastpath.guard`) gegen den Originalsatz: Fragen lösen keine
   Aktion aus, Zeitangaben werden im Code aus dem Satz gelesen, Playlists müssen im
   Satz vorkommen und werden gegen die bekannten Namen aufgelöst. Im Zweifel
   `unknown`.

Ergibt die Kette `unknown`, geht der Satz wie bisher an `ask.py`. Seltene Aktionen
(`timer_stop`, `spotify_now_playing`) erkennen nur die Regeln; das LLM bleibt bei
den sechs Kernaktionen.

Laufzeit am Spiegel: **llama-server** (wie `llm.service`), nicht Ollama.

## Begründung

| Variante (holdout, 26 Sätze) | Fehlaktionen | Treffer | Median | p90 |
|---|---|---|---|---|
| nur LLM, Prompt v1 (basis, 15 Sätze) | 2 von 15 | 87 % | 4,8 s | 7,9 s |
| Regeln + LLM v2 | 2 | 92 % | 1,6 s | 6,9 s |
| **Regeln + LLM v2 + Prüfschicht** | **0** | **100 %** | **1,6 s** | **6,9 s** |
| Regeln + LLM v3 (mehr Aktionen im Prompt) + Prüfschicht | 0 | 92 % | 1,6 s | 7,1 s |

- **Regeln zuerst:** Die Hälfte der Sätze ist eindeutig und braucht kein LLM
  (< 10 ms statt 3–7 s). Auf den unbekannten Holdout-Sätzen haben die Regeln
  11 von 11 richtig erkannt und keinen Satz falsch übernommen.
- **Rechnen gehört in den Code:** Alle Modelle machten aus „fünfundzwanzig“ 52
  (deutsche Zahlwörter nennen die Einer zuerst). `parse_minutes()` löst das
  vollständig.
- **Die Prüfschicht macht aus Fehlern harmlose Fehler:** Sie hat die beiden
  Playlist-Fehler von v2 korrigiert und blockiert erfundene Werte. Was übrig
  bleibt, endet als `unknown` — Jarvis fragt nach, statt falsch zu handeln.
- **1,5B ist auf dem Pi 4 die richtige Größe:** 0,5B ist schnell, hätte aber bei
  18 von 45 Befehlen (40 %) etwas Falsches ausgelöst (an/aus vertauscht, Playlists
  erfunden). 3B erreicht dieselbe Trefferquote bei doppelter Latenz und 60 °C.
- **Ein kleines Modell verträgt keinen überladenen Prompt:** v3 mit zwei weiteren
  Aktionen verschlechterte zwei vorher richtige Sätze. Neue Aktionen kommen daher
  in die Regeln, nicht in den Prompt.
- **llama-server statt Ollama:** Mit derselben GGUF-Datei waren beide gleich schnell
  (Median 1,59 s, p90 6,86 gegenüber 6,88 s, 3,6 tok/s) und lieferten Fall für Fall
  dieselben Ausgaben. Am Spiegel läuft llama-server bereits; ein zweites
  Laufzeitsystem brächte nur Betriebsaufwand.

## Konsequenzen
- **Latenz:** Regel-Treffer sofort, LLM-Fälle 3–7 s auf dem Pi 4. Jarvis sagt dabei
  wie bei freien Fragen „Moment.“. Per Software ist auf dem Pi 4 nichts mehr zu
  holen (llama-bench: 3,4 tok/s Erzeugung, 9 tok/s Prompt); schneller wird es nur
  mit mehr Regeln oder neuer Hardware (Pi 5: laut veröffentlichten Werten etwa
  2,5–3×).
- **RAM:** Qwen 2.5 1.5B belegt ~1,1 GB zusätzlich zu Gemma 3 1B (~0,8 GB).
  Offen: Ob Gemma 3 1B auch als Intent-Parser taugt (ein Modell, ein Dienst). Der
  Benchmark kann es direkt prüfen (`--backend llamacpp --url
  http://127.0.0.1:8081`).
- **Pflege:** `KNOWN_PLAYLISTS` muss die echten Playlists enthalten; später aus der
  Spotify-API laden. Neue Aktionen brauchen Wortschatz in den Regeln und Fälle im
  Selbsttest.
- **Aktionsnamen abgleichen:** Der Benchmark nutzt `spotify_*`, `lamp_*`, `timer_*`;
  `voice.py` kennt zusätzlich Fernseher, Gäste-WLAN und Lautstärke, aber noch keine
  Timer. Beim Einbau werden Schema und `INTENTS` vereinheitlicht.
- **Nachweis noch offen:** Prompt v2, Regeln und Prüfschicht sind mit Blick auf die
  Testsätze entstanden; auch `holdout.csv` ist nach dem Nachbessern „gesehen“. Der
  Status wechselt auf *angenommen*, wenn ein frischer Testsatz aus echten
  Vosk-Mitschriften (`bench/intent/record_testset.py`) keine Fehlaktionen zeigt.

## Verworfene Alternativen
- **Nur die Grammatik erweitern:** Jede neue Formulierung ist Handarbeit, und eine
  größere Grammatik erhöht die Verwechslungen in Vosk.
- **Nur LLM, ohne Regeln:** Median 3,75 s auch für „Licht an“.
- **Größeres Modell (3B, 7B):** gleiche Treffer bzw. zu langsam; die Fehler lagen in
  Zahlen und Sonderfällen, die Code zuverlässiger löst.
- **Ollama:** gleich schnell, aber ein zweiter Dienst neben `llm.service`.
- **Claude über AWS Bedrock für Befehle:** bessere Sprachkompetenz, aber der
  Befehlstext verließe das Haus und Jarvis wäre ohne Internet taub. Bleibt eine
  Option für freie Fragen (Kosten mit Haiku: wenige Cent im Monat).
- **Pi 5 sofort:** bringt nur Latenz, behebt keinen der gemessenen Fehler.
  Entscheidung nach dem frischen Testsatz; der Benchmark läuft dort unverändert.
