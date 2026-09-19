# ADR 0003 — MQTT als Event-Schnittstelle zu anderen Geräten

**Status:** vorgeschlagen · **Datum:** 2026-09

## Kontext
Ein zweites Projekt, ein MagicMirror mit Sensorerkennung, soll anzeigen, was
die Pipeline gerade erkennt, und darauf reagieren (z. B. Display einschalten).
Der Spiegel läuft auf einem anderen Gerät, in einer anderen Sprache (Node.js)
und hat einen eigenen Release-Zyklus.

## Entscheidung
Die Pipeline publiziert geschlossene Events und die aktuelle Präsenz über MQTT
(Mosquitto auf dem Edge-Pi). SQLite bleibt die einzige Quelle der Wahrheit;
MQTT ist ein Benachrichtigungskanal ohne Zustellgarantie über das hinaus, was
QoS 1 und die Queue im RAM leisten. Vertrag: [docs/mqtt.md](../mqtt.md).

## Begründung
- **Entkopplung:** Der Spiegel kennt nur Topics und ein versioniertes
  JSON-Schema, nicht Dateipfade, Tabellen oder Python-Code.
- **Push statt Poll:** Präsenz soll in Sekunden sichtbar sein, ohne dass der
  Spiegel die Datenbank abfragt.
- **Retained Messages + Last Will** liefern „Was ist gerade im Bild?“ und
  „Lebt die Pipeline?“ ohne eigenen Code auf der Broker-Seite.
- **Erweiterbar:** Weitere Sensoren oder ein Home-Assistant-Server können
  dieselben Topics abonnieren, ohne dass sich die Pipeline ändert.
- **Robust:** Fällt der Broker aus, läuft die Vision-Pipeline weiter — derselbe
  Grundsatz wie in ADR 0002 für die Reasoning-Ebene.

## Konsequenzen
- Neue Abhängigkeit `paho-mqtt` auf dem Pi, lazy importiert; ohne
  `--mqtt-host` läuft alles wie vorher.
- Sicherheit: anonym nur auf Loopback, im LAN Passwort und eine ACL, die dem
  Spiegel nur Lesen erlaubt. Zugangsdaten nur über Umgebungsvariablen bzw.
  Dateien auf dem Gerät, nie im Repo.
- Unverschlüsseltes MQTT im Heimnetz ist bewusst akzeptiert. Sobald Daten das
  LAN verlassen, ist TLS auf Port 8883 Pflicht.
- Schemaänderungen erhöhen `v`; Konsumenten ignorieren unbekannte Versionen.

## Verworfene Alternativen
- **Spiegel liest `events.db` direkt** (z. B. über einen Netzwerk-Mount):
  koppelt an das Tabellenschema, SQLite über Netzwerkdateisysteme ist
  unzuverlässig, und Präsenz steht gar nicht in der Datenbank.
- **HTTP-API auf dem Pi:** Der Spiegel müsste pollen; für Push bräuchte es
  WebSockets oder SSE — mehr eigener Code für dasselbe Ergebnis.
- **Home Assistant als Zwischenschicht:** sinnvoll später, als Voraussetzung
  aber zu schwer. Da Home Assistant MQTT spricht, bleibt der Weg offen.
