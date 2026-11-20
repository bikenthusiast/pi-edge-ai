#!/usr/bin/env python3
"""record_testset.py – frischen Testsatz für den Intent-Benchmark aufnehmen.

Anders als record_commands.py (Spiegel, feste Sätze) gibt es hier keine vorgegebenen
Sätze, sondern AUFGABENKARTEN: "Schalte das Licht aus – formuliere frei". Du sagst es
so, wie es dir in dem Moment einfällt; Vosk transkribiert OHNE Grammatik (genau wie
transcribe_free in voice.py). Was Vosk daraus macht, ist der Testsatz – Erkennungsfehler
inklusive. Die erwartete Aktion steht durch die Karte fest.

Damit ist der Satz wirklich unbekannt: weder Prompt noch Regeln noch Prüfschicht
wurden mit Blick auf ihn geschrieben. Bitte danach nichts nachbessern, bevor gemessen ist.

  uv run --with vosk record_testset.py --vosk-model ~/voice-models/vosk-model-small-de-0.15
  uv run --with vosk record_testset.py ... --takes 2          jede Karte 2×  (Standard: siehe CARDS)
  uv run --with vosk record_testset.py ... --list             Karten anzeigen
  python3 record_testset.py --dry-run                         ohne Mikrofon: Satz eintippen (Test)

Pro Aufnahme: Karte lesen, Enter, kurzer Countdown, sprechen (--seconds, Standard 4 s).
Danach siehst du die Transkription:  Enter = behalten,  r = nochmal,  d = verwerfen.
Die Transkription wird NICHT korrigiert – genau die Fehler sollen mitgemessen werden.

Ausgabe:
  <out>/vosk_<datum>.csv           Testsatz im Format von holdout.csv (quelle=vosk)
  <out>/vosk_<datum>/NNN_<aktion>.wav   die Aufnahmen (16 kHz mono), für spätere Vergleiche

Danach auf dem Pi (Ollama oder llama-server muss laufen):
  uv run --with httpx --with pydantic llm_bench.py run --label pi4 --prompts v2g \\
      --models qwen2.5:1.5b --fastpath --cases ~/voice-test/testsets/vosk_<datum>.csv

Gerät wie in record_commands.py: ReSpeaker Lite, 2 Kanäle, Kanal 0 = verarbeitetes Signal.
"""
from __future__ import annotations

import argparse
import array
import csv
import json
import os
import random
import subprocess
import sys
import time
import wave

RATE = 16000

# (Aktion, Kartentext, Anzahl Aufnahmen). Platzhalter:
#   {minuten}  -> zufällige Dauer; erwartet wird genau diese Zahl
#   playlist=? -> nach der Aufnahme fragt das Skript, welche Playlist du genannt hast
CARDS = [
    ("lamp_on", "Licht oder Lampe EINSCHALTEN", 3, {}),
    ("lamp_off", "Licht oder Lampe AUSSCHALTEN", 3, {}),
    ("spotify_play", "Musik / Spotify STARTEN (ohne Playlist)", 3, {}),
    ("spotify_play", "Die Playlist FEIERABEND starten", 2, {"playlist": "Feierabend"}),
    ("spotify_play", "Eine ANDERE Playlist starten (Name frei wählen)", 2, {"playlist": "?"}),
    ("spotify_pause", "Musik ANHALTEN / ausmachen", 3, {}),
    ("spotify_now_playing", "FRAGEN, welches Lied gerade läuft", 2, {}),
    ("timer_start", "Einen TIMER stellen: {minuten}", 4, {"minutes": "random"}),
    ("timer_stop", "Den laufenden TIMER STOPPEN", 2, {}),
    ("unknown", "Etwas sagen, das Jarvis NICHT kann (Wetter, Lautstärke, Uhrzeit, Smalltalk …)", 4, {}),
]

# Dauern mit typischen Stolpersteinen (Zahlwörter, Viertel-/halbe Stunde, Stunden)
MINUTES = [2, 5, 7, 12, 15, 20, 25, 30, 45, 58, 90]
MINUTES_SAY = {15: "15 Minuten (gern als „Viertelstunde“)", 30: "30 Minuten (gern als „halbe Stunde“)",
               45: "45 Minuten (gern als „Dreiviertelstunde“)", 90: "90 Minuten (gern als „eineinhalb Stunden“)"}

HEADER = """# Frischer Testsatz (Vosk-Mitschriften), aufgenommen mit record_testset.py.
# Sätze frei zu Aufgabenkarten gesprochen, Transkription NICHT korrigiert.
# NICHT zum Nachbessern von Prompt, Regeln oder Prüfschicht verwenden, bevor gemessen ist.
text;action;playlist;minutes;quelle;notiz
"""


def record(device: str, channels: int, channel: int, seconds: float) -> bytes:
    """arecord wie in record_commands.py; liefert 16-bit mono PCM des gewählten Kanals."""
    raw = subprocess.run(
        ["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(RATE), "-c", str(channels),
         "-d", str(int(seconds + 1)), "-t", "raw"],
        capture_output=True, check=True,
    ).stdout
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - len(raw) % (2 * channels)])
    mono = samples[channel::channels][: int(seconds * RATE)]
    return mono.tobytes()


def peak(pcm: bytes) -> int:
    a = array.array("h")
    a.frombytes(pcm)
    return max((abs(x) for x in a), default=0)


def save_wav(path: str, pcm: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def transcribe(rec, pcm: bytes) -> str:
    """Freie Vosk-Diktierung, wie transcribe_free() in voice.py."""
    rec.Reset()
    rec.AcceptWaveform(pcm)
    return json.loads(rec.FinalResult()).get("text", "")


def build_plan(takes: int | None) -> list[dict]:
    plan = []
    for action, text, n, extra in CARDS:
        for _ in range(takes or n):
            card = {"action": action, "text": text, "playlist": extra.get("playlist"), "minutes": None}
            if extra.get("minutes") == "random":
                m = random.choice(MINUTES)
                card["minutes"] = m
                card["text"] = text.format(minuten=MINUTES_SAY.get(m, f"{m} Minuten"))
            plan.append(card)
    random.shuffle(plan)  # keine Reihenfolge, die man sich merken könnte
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vosk-model", default=os.path.expanduser("~/voice-models/vosk-model-small-de-0.15"))
    p.add_argument("--out", default=os.path.expanduser("~/voice-test/testsets"))
    p.add_argument("--takes", type=int, help="Aufnahmen pro Karte (Standard: je Karte festgelegt)")
    p.add_argument("--seconds", type=float, default=4.0, help="Aufnahmefenster (freie Sätze sind länger)")
    p.add_argument("--device", default="plughw:CARD=Lite,DEV=0")
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--list", action="store_true", help="Karten anzeigen und beenden")
    p.add_argument("--dry-run", action="store_true", help="kein Mikrofon/Vosk: Satz eintippen")
    a = p.parse_args()

    if a.list:
        for action, text, n, _ in CARDS:
            print(f"{action:20} {a.takes or n}×  {text.format(minuten='<zufällige Dauer>')}")
        return

    rec = None
    if not a.dry_run:
        try:
            import vosk
        except ImportError:
            sys.exit("vosk fehlt: mit 'uv run --with vosk record_testset.py …' starten")
        if not os.path.isdir(a.vosk_model):
            sys.exit(f"Vosk-Modell nicht gefunden: {a.vosk_model}")
        vosk.SetLogLevel(-1)
        rec = vosk.KaldiRecognizer(vosk.Model(a.vosk_model), RATE)  # ohne Grammatik

    session = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(a.out, f"vosk_{session}.csv")
    wav_dir = os.path.join(a.out, f"vosk_{session}")
    os.makedirs(a.out, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(HEADER)

    plan = build_plan(a.takes)
    print(f"{len(plan)} Aufnahmen, je {a.seconds} s. Formuliere jedes Mal frei – nicht die Karte vorlesen.")
    print(f"Ctrl+C beendet, Behaltenes bleibt in {csv_path}\n")
    kept = 0
    for k, card in enumerate(plan, 1):
        while True:
            input(f"[{k}/{len(plan)}]  {card['text']}      (Enter)")
            if a.dry_run:
                pcm, text = b"", input("  Satz (wie Vosk ihn verstanden hätte): ").strip()
            else:
                print("  3.. 2.. 1.. jetzt", flush=True)
                time.sleep(0.6)
                pcm = record(a.device, a.channels, a.channel, a.seconds)
                text = transcribe(rec, pcm)
            level = "" if a.dry_run else f"   peak {peak(pcm)}"
            ans = input(f"  Vosk: „{text}“{level}   Enter=ok  r=nochmal  d=verwerfen: ").strip().lower()
            if ans == "r":
                continue
            if ans == "d" or not text:
                if not text and ans != "d":
                    print("  (nichts erkannt – verworfen, mit r direkt neu aufnehmen)")
                break
            playlist = card["playlist"]
            if playlist == "?":
                playlist = input("  Welche Playlist hast du genannt? (Schreibweise wie in Spotify): ").strip() or None
            wav_name = f"{k:03d}_{card['action']}.wav"
            if not a.dry_run:
                save_wav(os.path.join(wav_dir, wav_name), pcm)
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter=";").writerow([
                    text, card["action"], playlist or "", card["minutes"] or "", "vosk",
                    f"{card['text']} | {wav_name}",
                ])
            kept += 1
            break
    print(f"\n{kept} Sätze gespeichert: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbeendet")
