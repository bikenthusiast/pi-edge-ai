# ADR 0001 — LiteRT (`ai-edge-litert`) statt `tflite-runtime`

**Status:** akzeptiert · **Datum:** 2026-08

## Kontext
Der Inferenz-Layer für die Vision-Pipeline auf dem Pi 4 B musste gewählt werden.
Das System läuft auf Raspberry Pi OS Trixie mit Python 3.13.5 (aarch64).

## Entscheidung
`ai-edge-litert` als Interpreter-Paket.

## Begründung
- `tflite-runtime` wird nicht mehr gepflegt; letzte Wheels decken Python 3.13 nicht ab.
- `ai-edge-litert` liefert aarch64-Wheels für 3.10–3.14 → System-Python nutzbar,
  kein `uv`-Umweg wie bei MediaPipe (siehe ADR 0002).
- XNNPACK-Delegate ist per Default aktiv, ARM-optimiert.

## Konsequenzen
- Import lautet `from ai_edge_litert.interpreter import Interpreter`.
- 32-Bit-Images (armv7l) sind ausgeschlossen — es gibt keine Wheels.
- Tutorials von 2024/25 mit `tflite_runtime.interpreter` sind nicht 1:1 übertragbar.

## Verworfene Alternativen
- **ONNX Runtime:** funktioniert, aber größere Abhängigkeit ohne Mehrwert bei
  bereits quantisierten TFLite-Modellen.
- **Volles TensorFlow:** auf dem Pi 4 zu schwer, unnötig für reine Inferenz.
