# ADR 0004 — Gestenerkennung läuft in pi-edge-ai und wird als Edge-Event publiziert

**Status:** vorgeschlagen · **Datum:** 2026-10

## Kontext
Der MagicMirror soll per Handgeste reagieren, erster Anwendungsfall: Wischgeste
→ QR-Code für das Gäste-WLAN. Als Sensor ist ein DFRobot SEN0628 gewählt
(VL53L7CX, 8×8-Entfernungsmatrix, UART5), bewusst **ohne Kamera** wegen der
Privatsphäre (Entscheidung im MagicMirror-Projekt, 29.09.2026). Der Sensor hat
keine eingebaute Gestenfunktion; die 8×8-Frames müssen auf dem Pi klassifiziert
werden.

Offen war, in welches Repository Sensorauslese und Klassifikation gehören:
direkt in den MagicMirror (wie `presence.py` für den Radar) oder hierher.

## Entscheidung
Wahrnehmung hier, Reaktion im Spiegel:

- **pi-edge-ai** liest den Sensor, klassifiziert, schreibt jede Geste als Zeile
  in `events.db` und publiziert sie auf `edge/<device>/gesture`
  (Vertrag: [docs/mqtt.md](../mqtt.md), Schema v1).
- **raspberrypi-magicmirror** abonniert das Topic, bildet Gesten auf Aktionen ab
  und löst sie über seinen bestehenden Mechanismus aus (dort ADR-001 und ADR-008).

Frames kommen über eine gemeinsame Schnittstelle aus drei Quellen: synthetisch,
Aufzeichnung (JSONL) oder Sensor. Der erste Klassifikator ist regelbasiert
(`rules-v1`, Schwerpunktbahn der nahen Zonen).

## Begründung
- **Gleiche Aufgabe wie die Vision-Pipeline:** Sensordaten → Inferenz → ein
  Ereignis pro Episode → Eventlog + MQTT. Store, Publisher, Deploy-Grenze und
  hardwarefreie Tests existieren hier schon.
- **Entkopplung über einen Vertrag:** Der Spiegel kennt nur Topic und JSON, nicht
  die Klassifikation. Ein späterer gelernter Klassifikator (TinyML auf
  aufgezeichneten Frames) ändert für den Spiegel nichts außer `model`.
- **Testbar ohne Hardware:** Synthetische Frames und Aufzeichnungen erlauben den
  Durchstich Ende-zu-Ende, bevor der Sensor verkabelt ist.
- **Eigener Topic statt `events`:** Gesten sind Befehle, keine Beobachtungen.
  Konsumenten von `events` sollen nicht plötzlich Wischgesten interpretieren.

## Konsequenzen
- `gesture` ist **nicht retained**; Konsumenten verwerfen alte Nachrichten
  (QoS-1-Wiederzustellung nach Reconnect) und Duplikate über `id`.
- Gesten landen in derselben Tabelle `events` wie Bildklassifikationen,
  unterscheidbar über `source` (`sen0628`, `synthetic`, `replay`) und `model`.
- Die Beispielnutzlast `tests/fixtures/contract/gesture_v1.json` existiert in
  beiden Repositories und muss gemeinsam geändert werden.
- Heute laufen Pipeline und Spiegel auf **demselben** Pi 4B (siehe README,
  Benchmark mit MagicMirror). ADR 0003 nennt ein anderes Gerät; der Vertrag
  funktioniert in beiden Fällen, weil Host und Port beim Konsumenten
  konfigurierbar sind.
- Die Ausrichtung des Sensors im Rahmen wird über `--rotate`/`--mirror`
  kalibriert, nicht im Code.

## Verworfene Alternativen
- **Alles im MagicMirror-Repo** (Skript wie `presence.py`): weniger Teile, aber
  die Klassifikation wäre an den Spiegel gekoppelt, ohne Eventlog und ohne die
  vorhandene Testinfrastruktur.
- **Kamera mit Handerkennung (MediaPipe):** widerspricht der
  Privatsphäre-Entscheidung, kostet CPU (Inferenz halbiert sich bereits mit
  laufendem Spiegel) und wäre mit dem Sensor Wegwerfarbeit.
- **Gestencode auf dem RP2040 des Sensors:** möglich (ST STSW-IMG035), aber
  eigene Firmware ist ein eigenes Projekt; später prüfbar, der Vertrag bliebe.
