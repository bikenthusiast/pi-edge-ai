#!/usr/bin/env python3
"""LLM-Intent-Benchmark für Jarvis: mehrere Ollama-Modelle, gleiche Testfälle, Hardware-Vergleich.

Benchmark laufen lassen (auf dem Rechner, auf dem Ollama läuft):
    uv run --with httpx --with pydantic llm_bench.py run --label pi4
    uv run --with httpx --with pydantic llm_bench.py run --label pi4 --runs 3
    uv run --with httpx --with pydantic llm_bench.py run --label pi4 \
        --models qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b

Prompt-Varianten vergleichen (v1 = Basis, v2 = Beispiele + Zeitangaben im Code umrechnen,
v2g = v2 + Prüfschicht guard() aus fastpath.py,
v3 = v2 + timer_stop/spotify_now_playing im Prompt + Prüfschicht):
    uv run --with httpx --with pydantic llm_bench.py run --label pi4 --prompts v1 v2 \
        --models qwen2.5:1.5b qwen2.5:3b

Schneller Pfad (Regeln) vor dem LLM mitmessen (fastpath.py muss im selben Ordner liegen):
    uv run --with httpx --with pydantic llm_bench.py run --label pi4 --prompts v2 \
        --models qwen2.5:1.5b --fastpath

Eigene Testsätze aus CSV laden (z. B. Holdout-Satz, siehe holdout.csv):
    uv run --with httpx --with pydantic llm_bench.py run --label pi4 --prompts v2 \
        --models qwen2.5:1.5b --fastpath --cases holdout.csv

Zahlwort-Parser ohne Ollama prüfen:
    uv run --with httpx --with pydantic llm_bench.py selftest

Gleiches Modell über llama.cpp statt Ollama (llama-server muss laufen):
    uv run --with httpx --with pydantic llm_bench.py run --label pi4-llamacpp --backend llamacpp \
        --prompts v2g --models qwen2.5:1.5b --fastpath --cases holdout.csv

Läufe vergleichen (z. B. Pi 4 gegen Pi 5 oder Mac):
    uv run --with httpx --with pydantic llm_bench.py compare results/*.json
    uv run --with httpx --with pydantic llm_bench.py compare results/*.json --md vergleich.md

Ergebnisse landen als JSON in ./results/<label>_<zeitstempel>.json (Rohdaten je Testfall,
Zusammenfassung je Modell und Hardware-Metadaten).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

DEFAULT_URLS = {"ollama": "http://127.0.0.1:11434", "llamacpp": "http://127.0.0.1:8080"}
BACKEND = "ollama"  # wird in cmd_run aus --backend gesetzt
DEFAULT_MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]  # klein, mittel, groß


# --------------------------------------------------------------------------- Schema

class ActionCore(str, Enum):
    """Aktionen von Prompt v1/v2 (unverändert, damit alte Läufe vergleichbar bleiben)."""

    spotify_play = "spotify_play"
    spotify_pause = "spotify_pause"
    lamp_on = "lamp_on"
    lamp_off = "lamp_off"
    timer_start = "timer_start"
    unknown = "unknown"


class Action(str, Enum):
    """Alle Aktionen (ab Prompt v3)."""

    spotify_play = "spotify_play"
    spotify_pause = "spotify_pause"
    spotify_now_playing = "spotify_now_playing"
    lamp_on = "lamp_on"
    lamp_off = "lamp_off"
    timer_start = "timer_start"
    timer_stop = "timer_stop"
    unknown = "unknown"


class Intent(BaseModel):
    action: ActionCore
    playlist: str | None = Field(
        default=None, description="Name der Playlist, nur bei spotify_play, sonst null"
    )
    minutes: int | None = Field(
        default=None, ge=1, le=180, description="Dauer in Minuten, nur bei timer_start, sonst null"
    )


SYSTEM_PROMPT = """Du bist der Befehlsparser des Sprachassistenten Jarvis.
Der Eingabetext stammt aus einer Spracherkennung und kann Erkennungsfehler enthalten.
Ordne ihn genau einer Aktion zu und antworte ausschließlich mit JSON nach dem vorgegebenen Schema.

Aktionen:
- spotify_play: Musik starten oder fortsetzen; playlist nur setzen, wenn eine Playlist genannt wird
- spotify_pause: Musik anhalten oder stoppen
- lamp_on / lamp_off: Lampe bzw. Licht ein- oder ausschalten
- timer_start: Timer stellen; minutes = Dauer in Minuten (Zahlwörter in Ziffern umwandeln)
- unknown: alles andere
"""
SYSTEM_PROMPT_V1 = SYSTEM_PROMPT


class IntentV2(BaseModel):
    """v2: Das Modell rechnet nicht selbst, sondern gibt die Zeitangabe wörtlich zurück."""

    action: ActionCore
    playlist: str | None = Field(
        default=None, description="Name der Playlist wörtlich, nur bei spotify_play, sonst null"
    )
    duration: str | None = Field(
        default=None,
        description="Zeitangabe wörtlich aus dem Satz (z. B. 'zehn Minuten'), nur bei timer_start, sonst null",
    )


# Beispiele bewusst NICHT identisch mit den Testfällen, sonst misst der Test nur Auswendiglernen.
SYSTEM_PROMPT_V2 = """Du bist der Befehlsparser des Sprachassistenten Jarvis.
Der Eingabetext stammt aus einer Spracherkennung und kann Erkennungsfehler enthalten.
Ordne ihn genau einer Aktion zu und antworte ausschließlich mit JSON nach dem vorgegebenen Schema.

Aktionen:
- spotify_play: Musik starten oder fortsetzen; playlist nur setzen, wenn ausdrücklich eine Playlist genannt wird
- spotify_pause: Musik anhalten, stoppen, pausieren oder ausmachen
- lamp_on / lamp_off: Lampe bzw. Licht ein- oder ausschalten ("an" = ein, "aus" = aus)
- timer_start: Timer stellen; duration = die Zeitangabe wörtlich aus dem Satz, NICHT selbst umrechnen
- unknown: alles andere, auch Lautstärke, Wetter, Wissensfragen oder Befehle ohne passende Aktion

Regeln:
- Playlist-Namen genau so übernehmen, wie sie im Satz stehen.
- Wenn keine Aktion eindeutig passt, immer unknown. Niemals eine Playlist erfinden.

Beispiele:
"Musik bitte" -> {"action": "spotify_play"}
"Spiel die Playlist Gute Laune" -> {"action": "spotify_play", "playlist": "Gute Laune"}
"Kannst du mal die Playlist Roadtrip Klassiker anmachen" -> {"action": "spotify_play", "playlist": "Roadtrip Klassiker"}
"Stopp die Musik" -> {"action": "spotify_pause"}
"Mach das Licht aus" -> {"action": "lamp_off"}
"Schalt die Lampe ein" -> {"action": "lamp_on"}
"Timer auf eine halbe Stunde" -> {"action": "timer_start", "duration": "eine halbe Stunde"}
"Weck mich in dreiundvierzig Minuten" -> {"action": "timer_start", "duration": "dreiundvierzig Minuten"}
"Mach die Musik leiser" -> {"action": "unknown"}
"Was ist die Hauptstadt von Frankreich" -> {"action": "unknown"}
"""


class IntentV3(BaseModel):
    """v3: wie v2, plus timer_stop und spotify_now_playing; Antwort wird danach von guard() geprüft."""

    action: Action
    playlist: str | None = Field(
        default=None, description="Name der Playlist wörtlich, nur bei spotify_play, sonst null"
    )
    duration: str | None = Field(
        default=None,
        description="Zeitangabe wörtlich aus dem Satz (z. B. 'zehn Minuten'), nur bei timer_start, sonst null",
    )


SYSTEM_PROMPT_V3 = """Du bist der Befehlsparser des Sprachassistenten Jarvis.
Der Eingabetext stammt aus einer Spracherkennung und kann Erkennungsfehler enthalten.
Ordne ihn genau einer Aktion zu und antworte ausschließlich mit JSON nach dem vorgegebenen Schema.

Aktionen:
- spotify_play: Musik starten oder fortsetzen; playlist nur setzen, wenn ausdrücklich eine Playlist genannt wird
- spotify_pause: Musik anhalten, stoppen, pausieren oder ausmachen
- spotify_now_playing: Frage, welches Lied oder welche Musik gerade läuft
- lamp_on / lamp_off: Lampe bzw. Licht ein- oder ausschalten ("an" = ein, "aus" = aus)
- timer_start: Timer stellen; duration = die Zeitangabe wörtlich aus dem Satz, NICHT selbst umrechnen
- timer_stop: laufenden Timer stoppen, abbrechen oder löschen
- unknown: alles andere, auch Lautstärke, Wetter, Uhrzeit, Wissensfragen oder Befehle ohne passende Aktion

Regeln:
- Fragen sind keine Befehle: eine Frage ist spotify_now_playing oder unknown, niemals eine andere Aktion.
- Playlist-Namen vollständig und genau so übernehmen, wie sie im Satz stehen.
- Werte nur übernehmen, wenn sie im Satz vorkommen. Niemals etwas erfinden.
- Wenn keine Aktion eindeutig passt, immer unknown.

Beispiele:
"Musik bitte" -> {"action": "spotify_play"}
"Spiel die Playlist Gute Laune" -> {"action": "spotify_play", "playlist": "Gute Laune"}
"Mach mal die Playlist Musik zum Kochen an" -> {"action": "spotify_play", "playlist": "Musik zum Kochen"}
"Stopp die Musik" -> {"action": "spotify_pause"}
"Welcher Song ist das" -> {"action": "spotify_now_playing"}
"Mach das Licht aus" -> {"action": "lamp_off"}
"Schalt die Lampe ein" -> {"action": "lamp_on"}
"Timer auf eine halbe Stunde" -> {"action": "timer_start", "duration": "eine halbe Stunde"}
"Weck mich in dreiundvierzig Minuten" -> {"action": "timer_start", "duration": "dreiundvierzig Minuten"}
"Brich den Timer ab" -> {"action": "timer_stop"}
"Wie lange läuft der Timer noch" -> {"action": "unknown"}
"Mach die Musik leiser" -> {"action": "unknown"}
"Was ist die Hauptstadt von Frankreich" -> {"action": "unknown"}
"""


# Zeitangaben-Parser und schneller Pfad liegen in fastpath.py (gleicher Ordner)
import fastpath  # noqa: E402
from fastpath import parse_minutes  # noqa: E402


@dataclass(frozen=True)
class PromptVariant:
    schema: type[BaseModel]
    system: str
    to_final: object  # Callable[[BaseModel, str], tuple[dict, str | None]]  (Intent, Eingriff der Prüfschicht)


PROMPTS: dict[str, PromptVariant] = {
    "v1": PromptVariant(
        Intent,
        SYSTEM_PROMPT_V1,
        lambda i, text: ({"action": i.action.value, "playlist": i.playlist, "minutes": i.minutes}, None),
    ),
    "v2": PromptVariant(
        IntentV2,
        SYSTEM_PROMPT_V2,
        lambda i, text: ({"action": i.action.value, "playlist": i.playlist, "minutes": parse_minutes(i.duration)}, None),
    ),
    "v2g": PromptVariant(  # Prompt v2 unverändert, Antwort zusätzlich durch guard() geprüft
        IntentV2,
        SYSTEM_PROMPT_V2,
        lambda i, text: fastpath.guard(
            text, {"action": i.action.value, "playlist": i.playlist, "minutes": parse_minutes(i.duration)}
        ),
    ),
    "v3": PromptVariant(
        IntentV3,
        SYSTEM_PROMPT_V3,
        lambda i, text: fastpath.guard(
            text, {"action": i.action.value, "playlist": i.playlist, "minutes": parse_minutes(i.duration)}
        ),
    ),
}


# --------------------------------------------------------------------------- Testfälle

@dataclass(frozen=True)
class Case:
    text: str
    action: Action
    playlist: str | None = None
    minutes: int | None = None
    note: str = ""


CASES: list[Case] = [
    Case("Starte Spotify", Action.spotify_play),
    Case("Spiel mal die Playlist Feierabend", Action.spotify_play, playlist="Feierabend"),
    Case("Mach mal Musik von der Playlist Sonntagmorgen an", Action.spotify_play, playlist="Sonntagmorgen"),
    Case("spiel meine playlist deep focus", Action.spotify_play, playlist="Deep Focus", note="zweiteiliger Name"),
    Case("starte spotty fei", Action.spotify_play, note="Erkennungsfehler"),
    Case("Mach die Musik aus", Action.spotify_pause),
    Case("Pausier mal kurz die Musik", Action.spotify_pause),
    Case("Licht an", Action.lamp_on, note="sehr kurz"),
    Case("Lampe aus bitte", Action.lamp_off),
    Case("kannst du das licht ausmachen", Action.lamp_off, note="Frageform"),
    Case("Stell einen Timer auf zehn Minuten", Action.timer_start, minutes=10),
    Case("Timer für eine Viertelstunde", Action.timer_start, minutes=15, note="Umrechnung"),
    Case("stell mir nen timer auf fünfundzwanzig minuten", Action.timer_start, minutes=25, note="Zahlwort"),
    Case("Mach die Musik lauter", Action.unknown, note="Falle: keine Lautstärke-Aktion"),
    Case("Wie wird das Wetter morgen", Action.unknown),
]
WARMUP_TEXT = "Starte Spotify"


def load_cases(path: str) -> list[Case]:
    """CSV mit Semikolon: text;action;playlist;minutes[;quelle;notiz] – Zeilen mit # werden ignoriert."""
    import csv

    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    cases = []
    for row in csv.DictReader(lines, delimiter=";"):
        try:
            cases.append(Case(
                text=row["text"].strip(),
                action=Action(row["action"].strip()),
                playlist=(row.get("playlist") or "").strip() or None,
                minutes=int(row["minutes"]) if (row.get("minutes") or "").strip() else None,
                note=" ".join(x for x in [(row.get("quelle") or "").strip(), (row.get("notiz") or "").strip()] if x),
            ))
        except (KeyError, ValueError) as exc:
            sys.exit(f"Fehler in {path}, Zeile {row}: {exc}")
    if not cases:
        sys.exit(f"Keine Testfälle in {path} gefunden.")
    return cases


def _norm(value: str | None) -> str | None:
    return value.strip().casefold() if value else None


# Aktionen mit Wirkung in der echten Welt. Ein Fehler hier = "Fehlaktion".
# Harmlos sind Fehler, bei denen nichts passiert (unknown) oder nur etwas angesagt wird (now_playing).
SIDE_EFFECT_ACTIONS = {"spotify_play", "spotify_pause", "lamp_on", "lamp_off", "timer_start", "timer_stop"}


def is_harmful(rec: dict) -> bool:
    """Falsche Aktion mit Wirkung oder richtige Aktion mit falschem Wert (z. B. 52 statt 25 Minuten)."""
    got = rec.get("got")
    return (not rec["full_ok"]) and isinstance(got, dict) and got.get("action") in SIDE_EFFECT_ACTIONS


def check(final: dict, case: Case) -> dict:
    action = Action(getattr(final["action"], "value", final["action"]))
    action_ok = action == case.action
    full_ok = (
        action_ok
        and _norm(final["playlist"]) == _norm(case.playlist)
        and final["minutes"] == case.minutes
    )
    got = {"action": action.value}
    if final["playlist"] is not None:
        got["playlist"] = final["playlist"]
    if final["minutes"] is not None:
        got["minutes"] = final["minutes"]
    return {"valid": True, "action_ok": action_ok, "full_ok": full_ok, "got": got}


def evaluate(raw: str, case: Case, variant: PromptVariant) -> dict:
    try:
        parsed = variant.schema.model_validate_json(raw)
    except ValidationError:
        return {"valid": False, "action_ok": False, "full_ok": False, "got": raw[:200], "raw": raw[:200]}
    final, note = variant.to_final(parsed, case.text)
    return {**check(final, case), "guard": note, "raw": parsed.model_dump(mode="json", exclude_none=True)}


# --------------------------------------------------------------------------- Telemetrie

def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def read_temp() -> float | None:
    m = re.search(r"([\d.]+)", _cmd(["vcgencmd", "measure_temp"]))
    if m:
        return float(m.group(1))
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:
        return None


def read_clock_mhz() -> float | None:
    m = re.search(r"=(\d+)", _cmd(["vcgencmd", "measure_clock", "arm"]))
    if m:
        return int(m.group(1)) / 1e6
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text()) / 1000
    except Exception:
        return None


def read_throttled() -> int | None:
    m = re.search(r"0x([0-9a-fA-F]+)", _cmd(["vcgencmd", "get_throttled"]))
    return int(m.group(1), 16) if m else None


class Sampler:
    """Misst im Hintergrund Temperatur, Takt und Drosselung, solange ein Modell getestet wird."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.temps: list[float] = []
        self.clocks: list[float] = []
        self.throttled_now: int | None = None  # ODER der Bits 0-3 (aktuell gedrosselt/Unterspannung)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if (t := read_temp()) is not None:
                self.temps.append(t)
            if (c := read_clock_mhz()) is not None:
                self.clocks.append(c)
            if (th := read_throttled()) is not None:
                self.throttled_now = (self.throttled_now or 0) | (th & 0xF)
            self._stop.wait(self.interval)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        return {
            "temp_max": max(self.temps) if self.temps else None,
            "temp_mean": round(statistics.fmean(self.temps), 1) if self.temps else None,
            "clock_min_mhz": round(min(self.clocks)) if self.clocks else None,
            "clock_mean_mhz": round(statistics.fmean(self.clocks)) if self.clocks else None,
            "throttled_now": hex(self.throttled_now) if self.throttled_now is not None else "–",
        }


def hardware_info(client: httpx.Client, url: str) -> dict:
    model = ""
    dt = Path("/proc/device-tree/model")
    if dt.exists():
        model = dt.read_text(errors="ignore").strip("\x00\n ")
    elif sys.platform == "darwin":
        model = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem_gb = None
    try:
        kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
        mem_gb = round(kb / 1024**2, 1)
    except Exception:
        if out := _cmd(["sysctl", "-n", "hw.memsize"]):
            mem_gb = round(int(out) / 1024**3, 1)
    try:
        if BACKEND == "ollama":
            ollama_version = client.get(f"{url}/api/version").json().get("version")
        else:
            props = client.get(f"{url}/props").json()
            ollama_version = "llama.cpp " + str(props.get("build_info") or props.get("model_path", "")).strip()
    except Exception:
        ollama_version = "llama.cpp" if BACKEND == "llamacpp" else None
    return {
        "hostname": platform.node(),
        "model": model or platform.machine(),
        "cpu_count": os.cpu_count(),
        "mem_gb": mem_gb,
        "os": platform.platform(),
        "backend": BACKEND,
        "ollama_version": ollama_version,
        "throttled_at_start": hex(t) if (t := read_throttled()) is not None else None,
        "temp_at_start": read_temp(),
    }


# --------------------------------------------------------------------------- Backends (Ollama / llama.cpp)

def chat(client: httpx.Client, url: str, model: str, text: str, variant: PromptVariant) -> tuple[dict, float]:
    if BACKEND == "llamacpp":
        return chat_llamacpp(client, url, text, variant)
    payload = {
        "model": model,
        "stream": False,
        "format": variant.schema.model_json_schema(),
        "options": {"temperature": 0},
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": variant.system},
            {"role": "user", "content": text},
        ],
    }
    t0 = time.perf_counter()
    r = client.post(f"{url}/api/chat", json=payload)
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def chat_llamacpp(client: httpx.Client, url: str, text: str, variant: PromptVariant) -> tuple[dict, float]:
    """llama-server (OpenAI-kompatibel) mit JSON-Schema; Antwort ins Ollama-Format übersetzt."""
    payload = {
        "messages": [
            {"role": "system", "content": variant.system},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "cache_prompt": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "intent", "schema": variant.schema.model_json_schema()},
        },
    }
    t0 = time.perf_counter()
    r = client.post(f"{url}/v1/chat/completions", json=payload)
    r.raise_for_status()
    wall = time.perf_counter() - t0
    data = r.json()
    t = data.get("timings", {})
    return {
        "message": {"content": data["choices"][0]["message"]["content"]},
        "eval_count": t.get("predicted_n", 0),
        "eval_duration": t.get("predicted_ms", 0) * 1e6,
        "prompt_eval_count": t.get("prompt_n", 0),
        "prompt_eval_duration": t.get("prompt_ms", 0) * 1e6,
        "load_duration": 0,
    }, wall


def unload(client: httpx.Client, url: str, model: str) -> None:
    if BACKEND != "ollama":
        return
    try:
        client.post(f"{url}/api/generate", json={"model": model, "keep_alive": 0})
    except Exception:
        pass


def loaded_size_gb(client: httpx.Client, url: str, model: str) -> float | None:
    if BACKEND != "ollama":
        return None
    try:
        for m in client.get(f"{url}/api/ps").json().get("models", []):
            if model in (m.get("name"), m.get("model")):
                return round(m.get("size", 0) / 1024**3, 2)
    except Exception:
        pass
    return None


def installed_models(client: httpx.Client, url: str) -> set[str]:
    return {m["name"] for m in client.get(f"{url}/api/tags").json().get("models", [])}


def metrics(data: dict, wall: float) -> dict:
    eval_tokens = data.get("eval_count") or 0
    eval_s = (data.get("eval_duration") or 0) / 1e9
    return {
        "wall_s": round(wall, 3),
        "load_s": round((data.get("load_duration") or 0) / 1e9, 3),
        "prompt_tokens": data.get("prompt_eval_count") or 0,
        "prompt_s": round((data.get("prompt_eval_duration") or 0) / 1e9, 3),
        "eval_tokens": eval_tokens,
        "eval_s": round(eval_s, 3),
        "tok_s": round(eval_tokens / eval_s, 2) if eval_s else None,
    }


# --------------------------------------------------------------------------- run

def _p90(values: list[float]) -> float:
    return statistics.quantiles(values, n=10)[8] if len(values) >= 2 else values[0]


def summarize(model: str, prompt: str, use_fp: bool, cases_name: str, recs: list[dict], cold: dict,
              size_gb: float | None, tele: dict) -> dict:
    walls = [r["wall_s"] for r in recs]
    toks = [r["tok_s"] for r in recs if r["tok_s"]]
    llm = [r for r in recs if r.get("path", "llm") == "llm"] or recs
    n = len(recs)
    return {
        "model": model,
        "backend": BACKEND,
        "prompt": prompt,
        "fastpath": use_fp,
        "cases": cases_name,
        "fastpath_rate": round(sum(r.get("path") == "rules" for r in recs) / n, 3),
        "guard_count": sum(1 for r in recs if r.get("guard")),
        "harmful_count": sum(is_harmful(r) for r in recs),
        "n": n,
        "accuracy": round(sum(r["full_ok"] for r in recs) / n, 3),
        "action_accuracy": round(sum(r["action_ok"] for r in recs) / n, 3),
        "valid_rate": round(sum(r["valid"] for r in recs) / n, 3),
        "latency_median_s": round(statistics.median(walls), 2),
        "latency_p90_s": round(_p90(walls), 2),
        "tok_s_median": round(statistics.median(toks), 2) if toks else None,
        "latency_llm_median_s": round(statistics.median(r["wall_s"] for r in llm), 2),
        "prompt_s_median": round(statistics.median(r["prompt_s"] for r in llm), 2),
        "eval_tokens_median": statistics.median(r["eval_tokens"] for r in llm),
        "cold_start_s": round(cold["wall_s"], 2),
        "cold_load_s": round(cold["load_s"], 2),
        "model_ram_gb": size_gb,
        **tele,
    }


def cmd_run(args: argparse.Namespace) -> None:
    global BACKEND
    BACKEND = args.backend
    url = (args.url or DEFAULT_URLS[BACKEND]).rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        if BACKEND == "ollama":
            try:
                have = installed_models(client, url)
            except httpx.HTTPError as exc:
                sys.exit(f"Ollama unter {url} nicht erreichbar: {exc}")
            missing = [m for m in args.models if m not in have]
            if missing:
                sys.exit("Fehlende Modelle – zuerst holen:\n" + "\n".join(f"  ollama pull {m}" for m in missing))
        else:
            try:
                client.get(f"{url}/health").raise_for_status()
            except httpx.HTTPError as exc:
                sys.exit(f"llama-server unter {url} nicht erreichbar: {exc}")
            if len(args.models) != 1:
                sys.exit("Mit --backend llamacpp genau ein --models angeben (Name des Modells, das llama-server geladen hat).")

        cases = load_cases(args.cases) if args.cases else CASES
        cases_name = Path(args.cases).stem if args.cases else "basis"
        meta = hardware_info(client, url)
        print(f"Hardware: {meta['model']} | {meta['mem_gb']} GB | Backend {BACKEND} ({meta['ollama_version']})")
        print(f"Modelle: {', '.join(args.models)} | Testsatz {cases_name}: {len(cases)} Fälle × {args.runs} Durchläufe\n")

        records: list[dict] = []
        summaries: list[dict] = []
        for i, model in enumerate(args.models):
            for other in args.models:
                unload(client, url, other)
            if i > 0 and args.cooldown:
                print(f"… {args.cooldown} s Abkühlpause (aktuell {read_temp()} °C)")
                time.sleep(args.cooldown)

            for prompt_name in args.prompts:
                variant = PROMPTS[prompt_name]
                print(f"=== {model} | Prompt {prompt_name}{' + schneller Pfad' if args.fastpath else ''}")
                # Aufwärmen: beim ersten Prompt eines Modells inkl. Laden, sonst nur Prompt-Cache
                data, wall = chat(client, url, model, WARMUP_TEXT, variant)
                cold = metrics(data, wall)
                print(f"    Aufwärmen {cold['wall_s']:.1f} s (davon Laden {cold['load_s']:.1f} s)")

                model_recs: list[dict] = []
                with Sampler() as sampler:
                    for run in range(1, args.runs + 1):
                        for case in cases:
                            t0 = time.perf_counter()
                            hit = fastpath.match(case.text) if args.fastpath else None
                            if hit is not None:
                                result = {**check(hit, case), "raw": hit}
                                perf = {"wall_s": round(time.perf_counter() - t0, 4), "load_s": 0.0,
                                        "prompt_tokens": 0, "prompt_s": 0.0, "eval_tokens": 0,
                                        "eval_s": 0.0, "tok_s": None}
                                path = "rules"
                            else:
                                data, wall = chat(client, url, model, case.text, variant)
                                result = evaluate(data["message"]["content"], case, variant)
                                perf = metrics(data, wall)
                                path = "llm"
                            rec = {
                                "label": args.label,
                                "model": model,
                                "prompt": prompt_name,
                                "fastpath": args.fastpath,
                                "cases": cases_name,
                                "path": path,
                                "run": run,
                                "text": case.text,
                                "note": case.note,
                                "expected": {
                                    "action": case.action.value,
                                    "playlist": case.playlist,
                                    "minutes": case.minutes,
                                },
                                **result,
                                **perf,
                            }
                            model_recs.append(rec)
                            mark = "OK " if rec["full_ok"] else ("⚠  " if is_harmful(rec) else "~  ")
                            via = "Regel" if path == "rules" else "LLM  "
                            note = f"   [Prüfung: {rec['guard']}]" if rec.get("guard") else ""
                            print(f"    {mark} {via} {case.text!r:52} {rec['wall_s']:6.2f} s  -> {rec['got']}{note}")

                summary = summarize(
                    model, prompt_name, args.fastpath, cases_name, model_recs, cold, loaded_size_gb(client, url, model), sampler.summary()
                )
                summaries.append(summary)
                records.extend(model_recs)
                print(
                    f"    → Treffer {summary['accuracy']:.0%} | Fehlaktionen {summary['harmful_count']} | "
                    f"Regeln {summary['fastpath_rate']:.0%} | "
                    f"Median {summary['latency_median_s']} s | "
                    f"p90 {summary['latency_p90_s']} s | {summary['tok_s_median']} tok/s | "
                    f"Tmax {summary['temp_max']} °C | Drosselung {summary['throttled_now']}\n"
                )
            unload(client, url, model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{args.label}_{stamp}.json"
    result = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "meta": meta,
        "config": {
            "models": args.models,
            "prompts": args.prompts,
            "fastpath": args.fastpath,
            "known_playlists": fastpath.KNOWN_PLAYLISTS if args.fastpath else None,
            "runs": args.runs,
            "cases": cases_name,
            "n_cases": len(cases),
            "system_prompts": {p: PROMPTS[p].system for p in args.prompts},
        },
        "summary": summaries,
        "records": records,
    }
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(render_table([result]))
    print(f"\nGespeichert: {path}")


# --------------------------------------------------------------------------- compare

def _harmful(r: dict, s: dict) -> str:
    if "harmful_count" in s:
        return str(s["harmful_count"])
    recs = [x for x in r["records"] if x["model"] == s["model"] and _pl(x) == _pl(s)
            and x.get("cases", "basis") == s.get("cases", "basis")]
    return str(sum(is_harmful(x) for x in recs))


def _pl(s: dict) -> str:
    return s.get("prompt", "v1") + ("+fp" if s.get("fastpath") else "")


COLUMNS = [
    ("Hardware", lambda r, s: r["label"]),
    ("Modell", lambda r, s: s["model"] + (" [llama.cpp]" if s.get("backend") == "llamacpp" else "")),
    ("Testsatz", lambda r, s: s.get("cases", "basis")),
    ("Prompt", lambda r, s: _pl(s)),
    ("Regeln", lambda r, s: f"{s.get('fastpath_rate', 0):.0%}"),
    ("Prüfung", lambda r, s: str(s.get("guard_count", 0))),
    ("Fehlaktionen", _harmful),
    ("Treffer", lambda r, s: f"{s['accuracy']:.0%}"),
    ("Aktion", lambda r, s: f"{s['action_accuracy']:.0%}"),
    ("Median s", lambda r, s: f"{s['latency_median_s']:.2f}"),
    ("p90 s", lambda r, s: f"{s['latency_p90_s']:.2f}"),
    ("tok/s", lambda r, s: f"{s['tok_s_median']:.1f}" if s["tok_s_median"] else "–"),
    ("Kaltstart s", lambda r, s: f"{s['cold_start_s']:.1f}"),
    ("RAM GB", lambda r, s: f"{s['model_ram_gb']}" if s["model_ram_gb"] is not None else "–"),
    ("Tmax °C", lambda r, s: f"{s['temp_max']}" if s["temp_max"] is not None else "–"),
    ("Drosselung", lambda r, s: s["throttled_now"]),
]


def render_table(results: list[dict]) -> str:
    rows = [[fn(r, s) for _, fn in COLUMNS] for r in results for s in r["summary"]]
    header = [name for name, _ in COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_speedup(results: list[dict]) -> str:
    """Faktor gegenüber der ersten Datei (Basis), je Modell, das in mehreren Läufen vorkommt."""
    if len(results) < 2:
        return ""
    base = results[0]
    base_by_model = {(s["model"], _pl(s), s.get("cases", "basis")): s for s in base["summary"]}
    lines = [f"\nBeschleunigung gegenüber **{base['label']}** (Median-Latenz, >1 = schneller):", ""]
    lines += ["| Hardware | Modell | Prompt | Faktor Latenz | Faktor tok/s |", "|---|---|---|---|---|"]
    for r in results[1:]:
        for s in r["summary"]:
            b = base_by_model.get((s["model"], _pl(s), s.get("cases", "basis")))
            if not b:
                continue
            f_lat = b["latency_median_s"] / s["latency_median_s"] if s["latency_median_s"] else 0
            f_tok = (s["tok_s_median"] or 0) / b["tok_s_median"] if b["tok_s_median"] else 0
            lines.append(f"| {r['label']} | {s['model']} | {_pl(s)} | {f_lat:.2f}× | {f_tok:.2f}× |")
    return "\n".join(lines)


def render_failures(results: list[dict]) -> str:
    """Fehler je Lauf/Modell/Prompt, gleiche Fehler aus mehreren Durchläufen zusammengefasst."""
    counts: dict[tuple, int] = {}
    for r in results:
        for rec in r["records"]:
            if not rec["full_ok"]:
                key = (("⚠ " if is_harmful(rec) else "") + r["label"], rec["model"],
                       _pl(rec) + " / " + rec.get("cases", "basis"), rec["text"],
                       json.dumps(rec["got"], ensure_ascii=False) + (" [Regel]" if rec.get("path") == "rules" else ""),
                       json.dumps(rec["expected"], ensure_ascii=False))
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    lines = ["\nFehlerhafte Fälle (⚠ = Fehlaktion, hätte etwas Falsches ausgelöst):", ""]
    for (label, model, prompt, text, got, exp), n in counts.items():
        lines.append(f"- {label} / {model} / {prompt}: {text!r} → {got} (erwartet {exp}) ×{n}")
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> None:
    results = [json.loads(Path(p).read_text()) for p in args.files]
    hw = "\n".join(
        f"- **{r['label']}**: {r['meta']['model']}, {r['meta']['mem_gb']} GB, "
        f"{r['meta'].get('backend', 'ollama')} {r['meta']['ollama_version']}, {r['timestamp']}"
        for r in results
    )
    text = "\n".join(
        part for part in [f"## Hardware\n\n{hw}\n\n## Ergebnisse\n", render_table(results),
                          render_speedup(results), render_failures(results)] if part
    )
    print(text)
    if args.md:
        Path(args.md).write_text("# LLM-Intent-Benchmark\n\n" + text + "\n")
        print(f"\nGespeichert: {args.md}")


# --------------------------------------------------------------------------- selftest

def cmd_selftest(_: argparse.Namespace) -> None:
    sys.exit(1 if fastpath.selftest() else 0)


# --------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Benchmark ausführen")
    run.add_argument("--label", required=True, help="Name der Hardware, z. B. pi4, pi5, mac")
    run.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    run.add_argument("--prompts", nargs="+", default=["v1"], choices=sorted(PROMPTS),
                     help="Prompt-Varianten (Standard v1)")
    run.add_argument("--fastpath", action="store_true",
                     help="Eindeutige Befehle zuerst per Regeln (fastpath.py), nur der Rest ans LLM")
    run.add_argument("--cases", help="Testsätze aus CSV statt der eingebauten 15 (z. B. holdout.csv)")
    run.add_argument("--runs", type=int, default=1, help="Durchläufe pro Testfall (Standard 1)")
    run.add_argument("--cooldown", type=int, default=60, help="Abkühlpause zwischen Modellen in s")
    run.add_argument("--backend", choices=["ollama", "llamacpp"], default="ollama",
                     help="ollama (Standard) oder llamacpp (llama-server, OpenAI-kompatibel)")
    run.add_argument("--url", help="Standard: Ollama :11434, llama-server :8080")
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--out", default="results")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="Ergebnisdateien vergleichen (erste Datei = Basis)")
    cmp_.add_argument("files", nargs="+")
    cmp_.add_argument("--md", help="Vergleich zusätzlich als Markdown-Datei speichern")
    cmp_.set_defaults(func=cmd_compare)

    st = sub.add_parser("selftest", help="Schnellen Pfad und Zeitangaben-Parser prüfen (ohne Ollama)")
    st.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
