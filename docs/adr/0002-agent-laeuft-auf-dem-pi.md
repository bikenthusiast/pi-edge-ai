# ADR 0002 — Der Agent läuft auf dem Pi, nicht in Lambda

**Status:** akzeptiert · **Datum:** 2026-08

## Kontext
Der Strands-Agent des Edge Ops Copilot könnte auf dem Pi oder cloudseitig
(Lambda / Bedrock AgentCore Runtime) laufen.

## Entscheidung
Agent-Loop auf dem Pi; nur die Modell-Inferenz und das Retrieval gehen zu AWS.

## Begründung
- Tools brauchen lokalen Zugriff: GPIO, Kamera, SQLite-Eventlog.
- Ein cloudseitiger Agent müsste den Pi über eine eingehende Verbindung
  erreichen — zusätzliche Angriffsfläche, NAT-Probleme.
- Der Edge/Cloud-Split ist das eigentliche Architekturargument des Projekts.

## Konsequenzen
- Der Pi braucht AWS-Credentials → eigener IAM-User mit minimalen Rechten,
  Rotation dokumentiert in `infra/iam/`.
- Bei Netzausfall bleibt die Vision-Pipeline funktionsfähig, nur die
  Reasoning-Ebene fällt aus. Das ist gewollt und wird getestet.

## Verworfene Alternativen
- **Bedrock Agents Classic:** seit 30.07.2026 für Neukunden geschlossen.
- **Alles in Lambda:** verliert den Edge-Aspekt, erzwingt Frame-Upload.
