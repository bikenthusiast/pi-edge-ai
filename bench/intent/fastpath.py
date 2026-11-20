#!/usr/bin/env python3
"""Schneller Pfad für Jarvis: eindeutige Sprachbefehle per Regeln erkennen, ohne LLM.

Prinzip: Präzision vor Abdeckung. Ein Satz wird nur erkannt, wenn nach dem Entfernen von
Füllwörtern ALLE Wörter zum Wortschatz genau einer Aktion gehören. Alles andere ergibt None,
und dann entscheidet das LLM. Lieber einmal zu oft ans LLM als einmal die falsche Aktion.

Zweite Aufgabe: guard() prüft jede LLM-Antwort gegen den Originalsatz, bevor Jarvis handelt
(Fragen lösen keine Aktion aus, Werte müssen im Satz vorkommen, Playlists werden aufgelöst).

Verwendung:
    from fastpath import match, guard
    match("Mach das Licht aus")        # -> {"action": "lamp_off", "playlist": None, "minutes": None}
    match("Mach die Musik lauter")     # -> None (unbekannt -> LLM)
    guard("Stopp den Timer", {"action": "timer_start", "playlist": None, "minutes": 1})
                                       # -> ({"action": "unknown", ...}, "Dauer nicht im Satz")

Selbsttest (ohne Abhängigkeiten):
    python3 fastpath.py
"""
from __future__ import annotations

import re

# Eigene Playlists hier eintragen (Schreibweise = wie an Spotify übergeben).
KNOWN_PLAYLISTS: list[str] = ["Feierabend"]

# --------------------------------------------------------------------------- Zahlen & Zeitangaben

_UNITS = {
    "null": 0, "ein": 1, "eins": 1, "eine": 1, "einen": 1, "einer": 1, "zwei": 2, "zwo": 2,
    "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
}
_TEENS = {
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
}
_TENS = {
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
    "siebzig": 70, "achtzig": 80, "neunzig": 90,
}


def german_number(word: str) -> int | None:
    """'fünfundzwanzig' -> 25, 'hundertzwanzig' -> 120, '10' -> 10, sonst None."""
    if word.isdigit():
        return int(word)
    w = word.lower().replace("ue", "ü").replace("ss", "ß")
    if "hundert" in w:
        left, right = w.split("hundert", 1)
        h = german_number(left) if left else 1
        r = german_number(right) if right else 0
        return None if h is None or r is None else h * 100 + r
    for table in (_UNITS, _TEENS, _TENS):
        if w in table:
            return table[w]
    if "und" in w:
        a, b = w.split("und", 1)
        if a in _UNITS and b in _TENS:
            return _UNITS[a] + _TENS[b]
    return None


def parse_minutes(text: str | None) -> int | None:
    """Deutsche Zeitangabe -> Minuten. 'eine Viertelstunde' -> 15, 'eine Stunde und zehn Minuten' -> 70."""
    if not text:
        return None
    t = text.lower().replace("-", " ")
    if "dreiviertel" in t:
        return 45
    if re.search(r"viertel\s*stunde", t):
        return 15
    if re.search(r"\bhalbe?n?\s*stunde", t):
        return 30
    if "anderthalb" in t or "eineinhalb" in t:
        return 90
    total, pending, found = 0, None, False
    for tok in re.findall(r"\d+|[a-zäöüß]+", t):
        if tok.startswith("stunde"):
            total += (pending if pending is not None else 1) * 60
            pending, found = None, True
        elif tok.startswith("minute") or tok == "min":
            if pending is not None:
                total += pending
                pending, found = None, True
        elif (n := german_number(tok)) is not None:
            pending = n
    if pending is not None:  # Zahl ohne Einheit -> Minuten
        total += pending
        found = True
    return total if found and total > 0 else None


# --------------------------------------------------------------------------- Wortschatz

FILLER = {
    "bitte", "mal", "doch", "jetzt", "kurz", "schnell", "gerade", "wieder", "mir", "nen", "einen",
    "eine", "die", "das", "den", "der", "dem", "du", "kannst", "könntest", "würdest", "kann", "jarvis",
    "hey", "hallo", "okay", "ok", "mach", "mache", "machen", "schalt", "schalte", "schalten", "meine",
    "mein", "noch", "danke",
}
NEGATION = {"nicht", "kein", "keine", "nie", "niemals"}

LAMP = {"licht", "lampe", "leuchte", "beleuchtung"}
LAMP_ON = {"an", "ein", "anmachen", "einschalten", "anschalten", "anknipsen"}
LAMP_OFF = {"aus", "ausmachen", "ausschalten", "abschalten", "ausknipsen"}

MUSIC = {"musik", "spotify", "song", "lied", "wiedergabe"}
PAUSE = {
    "aus", "ausmachen", "ausschalten", "stopp", "stop", "stoppe", "stoppen", "pause", "pausieren",
    "pausier", "pausiere", "anhalten", "halt", "beenden",
}
PLAY = {
    "starte", "start", "starten", "spiel", "spiele", "spielen", "abspielen", "an", "anmachen",
    "weiter", "fortsetzen", "los",
}
PLAYLIST_WORDS = {"playlist", "playliste", "liste", "wiedergabeliste"}

TIMER = {"timer", "wecker", "countdown"}
TIMER_STOP = {
    "stopp", "stop", "stoppe", "stoppen", "aus", "ausmachen", "ausschalten", "abbrechen", "brich", "ab",
    "beenden", "beende", "löschen", "lösche", "weg", "anhalten", "halt",
}
QUESTION_WORDS = {
    "was", "wie", "wer", "wann", "wo", "welche", "welcher", "welches", "welchen", "warum", "wieso",
    "weshalb", "wieviel", "wohin", "woher",
}
NOW_PLAYING_CORE = {"läuft", "spielt", "lied", "song", "titel"}
NOW_PLAYING_WORDS = NOW_PLAYING_CORE | {
    "was", "welches", "welcher", "musik", "spotify", "ist", "für", "ein", "hier", "auf", "da", "grad",
}
TIMER_WORDS = {
    "stell", "stelle", "stellen", "setz", "setze", "starte", "start", "auf", "für", "in", "von",
    "minuten", "minute", "min", "stunde", "stunden", "und", "halbe", "halben", "viertelstunde",
    "dreiviertelstunde", "eineinhalb", "anderthalb",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zäöüß0-9]+", text.lower())


def is_question(text: str) -> bool:
    toks = [t for t in _tokens(text) if t not in FILLER]
    return bool(toks) and (toks[0] in QUESTION_WORDS or text.strip().endswith("?"))


def _intent(action: str, playlist: str | None = None, minutes: int | None = None) -> dict:
    return {"action": action, "playlist": playlist, "minutes": minutes}


def match(text: str, playlists: list[str] | None = None) -> dict | None:
    """Erkennt eindeutige Befehle. Gibt ein Intent-Dict zurück oder None (-> LLM)."""
    all_toks = _tokens(text)
    toks = [t for t in all_toks if t not in FILLER]
    words = set(toks)
    if not words or words & NEGATION:
        return None

    # Fragen: nur "Was läuft gerade?" wird erkannt, alles andere -> LLM
    if is_question(text):
        if words <= NOW_PLAYING_WORDS and words & NOW_PLAYING_CORE:
            return _intent("spotify_now_playing")
        return None

    # Lampe: Objekt + genau eine Richtung, sonst nichts
    if words & LAMP:
        rest = words - LAMP
        if rest and rest <= LAMP_ON:
            return _intent("lamp_on")
        if rest and rest <= LAMP_OFF:
            return _intent("lamp_off")
        return None

    # Bekannte Playlist (auch mehrteilige Namen)
    joined = " " + " ".join(all_toks) + " "
    for name in playlists if playlists is not None else KNOWN_PLAYLISTS:
        if f" {name.lower()} " in joined:
            rest = words - set(_tokens(name)) - PLAY - MUSIC - PLAYLIST_WORDS - {"von"}
            return _intent("spotify_play", playlist=name) if not rest else None

    # Musik ohne Playlist: Objekt + nur Start- oder nur Stopp-Wörter
    if words & MUSIC and not words & PLAYLIST_WORDS:
        rest = words - MUSIC
        if rest and rest <= PAUSE:
            return _intent("spotify_pause")
        if rest and rest <= PLAY:
            return _intent("spotify_play")
        return None

    # Timer: Objekt + Zeitangabe, sonst nur Timer-Wörter und Zahlen
    if words & TIMER:
        rest = words - TIMER
        if rest and rest <= TIMER_STOP:
            return _intent("timer_stop")
        minutes = parse_minutes(" ".join(toks))
        rest = {w for w in words - TIMER - TIMER_WORDS if german_number(w) is None}
        if minutes and not rest:
            return _intent("timer_start", minutes=minutes)
        return None

    return None


# --------------------------------------------------------------------------- Prüfschicht für LLM-Antworten

PLAYLIST_MARKERS = {"playlist", "playliste", "wiedergabeliste", "liste"}
TRAILING = {"ab", "an", "auf", "abspielen", "spielen", "starten", "laufen", "bitte", "lassen"}
_NOT_GROUNDED = object()


def _squash(value: str) -> str:
    return re.sub(r"[^a-zäöüß0-9]", "", value.lower())


def resolve_playlist(name: str | None, text: str, playlists: list[str] | None = None):
    """Playlist aus LLM-Ausgabe und Originalsatz bestimmen.

    1. Bekannte Playlist, unabhängig von Leerzeichen ('feier abend' -> 'Feierabend').
    2. Name muss im Satz vorkommen, sonst _NOT_GROUNDED (erfunden).
    3. Abgeschnittene Namen nach dem Wort 'Playlist' ergänzen ('Chill' -> 'Chill am Abend').
    """
    sq_text, sq_name = _squash(text), _squash(name or "")
    for known in sorted(playlists if playlists is not None else KNOWN_PLAYLISTS, key=len, reverse=True):
        k = _squash(known)
        if k and (k in sq_text or (sq_name and k in sq_name)):
            return known
    if sq_name and sq_name not in sq_text:
        return _NOT_GROUNDED
    words = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text)
    for i, w in enumerate(words):
        if w.lower() in PLAYLIST_MARKERS:
            tail = words[i + 1:]
            while tail and tail[-1].lower() in TRAILING:
                tail.pop()
            candidate = " ".join(tail)
            if candidate and _squash(candidate).startswith(sq_name):
                return candidate
    return name


def guard(text: str, intent: dict, playlists: list[str] | None = None) -> tuple[dict, str | None]:
    """Prüft eine LLM-Antwort gegen den Originalsatz. Gibt (Intent, Grund des Eingriffs oder None) zurück."""
    action = intent["action"]
    unknown = _intent("unknown")

    if is_question(text) and action not in ("spotify_now_playing", "unknown"):
        return unknown, f"Frage löst keine Aktion aus ({action})"

    if action == "spotify_play":
        resolved = resolve_playlist(intent.get("playlist"), text, playlists)
        if resolved is _NOT_GROUNDED:
            return unknown, f"Playlist {intent.get('playlist')!r} kommt im Satz nicht vor"
        if resolved != intent.get("playlist"):
            return _intent("spotify_play", playlist=resolved), f"Playlist {intent.get('playlist')!r} -> {resolved!r}"
        return _intent("spotify_play", playlist=resolved), None

    if action == "timer_start":
        from_text = parse_minutes(text)
        if from_text is None:
            return unknown, "Timer ohne Zeitangabe im Satz"
        if intent.get("minutes") not in (None, from_text):  # Zahl aus dem Satz ist verlässlicher
            return _intent("timer_start", minutes=from_text), f"Dauer {intent.get('minutes')} -> {from_text} (aus dem Satz)"
        return _intent("timer_start", minutes=from_text), None

    # Aktionen ohne Werte: Werte verwerfen
    return _intent(action), None


# --------------------------------------------------------------------------- Selbsttest

_MINUTES_TESTS = [
    ("zehn Minuten", 10), ("fünfundzwanzig Minuten", 25), ("fuenfundzwanzig minuten", 25),
    ("eine Viertelstunde", 15), ("eine halbe Stunde", 30), ("Dreiviertelstunde", 45),
    ("eineinhalb Stunden", 90), ("eine Stunde", 60), ("zwei Stunden", 120),
    ("eine Stunde und zehn Minuten", 70), ("45 Minuten", 45), ("dreiundvierzig", 43),
    ("hundertzwanzig Minuten", 120), ("zwölf Minuten", 12), ("Minuten", None), (None, None),
]

# (Satz, erwartete Aktion oder None = muss ans LLM gehen, Playlist, Minuten)
_MATCH_TESTS = [
    ("Licht an", "lamp_on", None, None),
    ("Licht aus", "lamp_off", None, None),
    ("Schalt die Lampe ein", "lamp_on", None, None),
    ("kannst du das licht ausmachen", "lamp_off", None, None),
    ("Mach die Musik aus", "spotify_pause", None, None),
    ("Musik stopp", "spotify_pause", None, None),
    ("Spotify pausieren", "spotify_pause", None, None),
    ("Starte Spotify", "spotify_play", None, None),
    ("Musik an", "spotify_play", None, None),
    ("Spiel mal die Playlist Feierabend", "spotify_play", "Feierabend", None),
    ("Feierabend Playlist abspielen", "spotify_play", "Feierabend", None),
    ("Stell einen Timer auf zehn Minuten", "timer_start", None, 10),
    ("Timer für eine Viertelstunde", "timer_start", None, 15),
    ("Wecker auf eine Stunde", "timer_start", None, 60),
    ("Timer 5 Minuten", "timer_start", None, 5),
    ("Brich den Timer ab", "timer_stop", None, None),
    ("Timer aus", "timer_stop", None, None),
    ("Welches Lied ist das", "spotify_now_playing", None, None),
    ("Jarvis, was spielt da gerade?", "spotify_now_playing", None, None),
    # Diese MÜSSEN ans LLM gehen (None):
    ("Mach die Musik lauter", None, None, None),
    ("Wie wird das Wetter morgen", None, None, None),
    ("Lass das Licht nicht an", None, None, None),
    ("Mach das Licht im Bad an", None, None, None),
    ("Licht an und Musik aus", None, None, None),
    ("Spiel Musik von Queen", None, None, None),
    ("Spiel meine Playlist Deep Focus", None, None, None),
    ("Erzähl mir was über Spotify", None, None, None),
    ("starte spotty fei", None, None, None),
    ("Stell einen Timer", None, None, None),
    ("Timer auf ein paar Minuten", None, None, None),
    ("Licht", None, None, None),
    ("Musik", None, None, None),
    ("Wie lange läuft der Timer noch", None, None, None),
    ("Was ist mit dem Licht", None, None, None),
]

# (Satz, LLM-Antwort, erwartetes Ergebnis nach guard)
_GUARD_TESTS = [
    ("Stopp den Timer", ("timer_start", None, 1), ("unknown", None, None)),
    ("Was läuft gerade auf Spotify", ("spotify_play", None, None), ("unknown", None, None)),
    ("spiel die play liste feier abend", ("spotify_play", "Play Liste Feier Abend", None), ("spotify_play", "Feierabend", None)),
    ("Spiel meine Playlist Chill am Abend", ("spotify_play", "Chill", None), ("spotify_play", "Chill am Abend", None)),
    ("Leg die Liste Sommerhits auf", ("spotify_play", None, None), ("spotify_play", "Sommerhits", None)),
    ("Mach die Musik lauter", ("spotify_play", "Laufplaylist", None), ("unknown", None, None)),
    ("Erinnere mich in zwanzig Minuten", ("timer_start", None, 20), ("timer_start", None, 20)),
    ("stell mir nen timer auf fünfundzwanzig minuten", ("timer_start", None, 52), ("timer_start", None, 25)),
    ("Eieruhr auf sieben Minuten", ("timer_start", None, None), ("timer_start", None, 7)),
    ("Mach es hell", ("lamp_on", "x", 3), ("lamp_on", None, None)),
]


def selftest() -> int:
    bad = 0
    print("parse_minutes:")
    for text, expected in _MINUTES_TESTS:
        got = parse_minutes(text)
        bad += got != expected
        print(f"  {'OK' if got == expected else 'XX'}  {text!r:36} -> {got}  (erwartet {expected})")
    print("\nmatch:")
    for text, action, playlist, minutes in _MATCH_TESTS:
        got = match(text)
        expected = _intent(action, playlist, minutes) if action else None
        ok = got == expected
        bad += not ok
        shown = "→ LLM" if got is None else got["action"] + (f" {got['playlist']}" if got["playlist"] else "") + (
            f" {got['minutes']} min" if got["minutes"] else "")
        print(f"  {'OK' if ok else 'XX'}  {text!r:40} {shown}")
    print("\nguard:")
    for text, llm, expected in _GUARD_TESTS:
        got, note = guard(text, _intent(*llm))
        ok = got == _intent(*expected)
        bad += not ok
        print(f"  {'OK' if ok else 'XX'}  {text!r:46} {llm[0]} -> {got['action']}"
              + (f" {got['playlist']}" if got["playlist"] else "") + (f" {got['minutes']} min" if got["minutes"] else "")
              + (f"   [{note}]" if note else ""))
    print(f"\n{'Alle Tests bestanden.' if not bad else f'{bad} Fehler.'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if selftest() else 0)
