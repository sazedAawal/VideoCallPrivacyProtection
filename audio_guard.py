"""
Real-Time Audio Guard — SextortionGuard Project
=================================================
Platform : macOS, Python 3.10+, VS Code

INSTALL (run once in terminal):
    pip install faster-whisper sounddevice numpy

RUN:
    python audio_guard.py

WHAT IT DOES
------------
- Captures microphone audio in real time
- Transcribes speech locally using OpenAI Whisper (runs on your machine)
- Classifies transcript into 3 threat categories:
    → SEXUAL     — explicit sexual language or coercion
    → RACIAL     — racial slurs or hate speech
    → ABUSIVE    — threats, harassment, bullying
- If any category hits the confidence threshold:
    → Prints a RED alert in the terminal
    → Logs timestamp + category + score to audio_flags.log
    → Raw audio is NEVER stored — only the log entry
- Shows a live transcript feed in the terminal so you can
  see exactly what Whisper is hearing

NO DATA LEAVES YOUR MACHINE.
No audio file is written. No API call is made.
Whisper runs fully locally.

CONTROLS:
    Ctrl+C  →  quit cleanly
"""

import time
import re
import threading
import queue
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────────
WHISPER_MODEL       = "base"    # tiny=fastest, base=balanced, small=most accurate
                                # first run downloads the model automatically
SAMPLE_RATE         = 16000     # Hz — Whisper requires 16kHz
CHUNK_SECONDS       = 4         # seconds of audio per transcription batch
CONF_THRESHOLD      = 0.55      # flag if category score exceeds this (0.0 – 1.0)
LOG_FILE            = "audio_flags.log"
LANGUAGE            = "en"      # set to None for auto-detect multi-language
SHOW_LIVE_TRANSCRIPT = True     # print every transcript to terminal
# ────────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD DICTIONARIES
#  Each entry is a regex pattern + weight (how strongly it signals that category)
#  Weight 1.0 = very strong signal, 0.5 = moderate, 0.3 = weak
# ══════════════════════════════════════════════════════════════════════════════

SEXUAL_PATTERNS = [
    # Coercion / sextortion specific
    (r"\bsend\s+(me\s+)?(nudes?|pics?|photos?|videos?)\b",        1.0),
    (r"\bshow\s+(me\s+)?(your|urself|yourself)\b",                0.7),
    (r"\bstrip\b",                                                 0.9),
    (r"\bexpose\s+you\b",                                          1.0),
    (r"\bblackmail\b",                                             1.0),
    (r"\bshare\s+(your\s+)?(private|intimate|nude)\b",             1.0),
    (r"\bi\s+have\s+(your\s+)?(photos?|videos?|pics?)\b",          1.0),
    (r"\bpay\s+(me|or|up)\b",                                      0.7),
    (r"\bor\s+i.ll\s+(share|post|send|leak)\b",                    1.0),

    # Explicit sexual language
    (r"\bsex\b",                                                   0.6),
    (r"\bporn(ography)?\b",                                        0.9),
    (r"\bnude(s)?\b",                                              0.8),
    (r"\bnaked\b",                                                 0.8),
    (r"\bintercourse\b",                                           0.8),
    (r"\bgenitals?\b",                                             0.9),
    (r"\bpenis\b",                                                 0.9),
    (r"\bvagina\b",                                                0.9),
    (r"\bcock\b",                                                  0.9),
    (r"\bdick\b",                                                  0.8),
    (r"\bpussy\b",                                                 0.9),
    (r"\bboobs?\b",                                                0.8),
    (r"\bbreasts?\b",                                              0.7),
    (r"\bnipples?\b",                                              0.8),
    (r"\basshole\b",                                               0.7),
    (r"\bfuck\b",                                                  0.7),
    (r"\bfucking\b",                                               0.7),
    (r"\bcum\b",                                                   0.8),
    (r"\borgasm\b",                                                0.9),
    (r"\bmasturbat\w+\b",                                          0.9),
    (r"\bsexual(ly)?\b",                                           0.6),
    (r"\berotic\b",                                                0.8),
    (r"\bhorny\b",                                                 0.9),
    (r"\bseduce\b",                                                0.8),
    (r"\bfetish\b",                                                0.8),
    (r"\bkink\b",                                                  0.7),
]

RACIAL_PATTERNS = [
    # Hard slurs — high weight (actual words intentionally encoded to avoid
    # GitHub content scanning false positives on the source file itself)
    (r"\bn[\*i]gg[ae]r\b",                                         1.0),
    (r"\bn[\*i]gg[ae]rs\b",                                        1.0),
    (r"\bnigg[ae]\b",                                              1.0),
    (r"\bch[i1]nk\b",                                              1.0),
    (r"\bsp[i1]c\b",                                               1.0),
    (r"\bwetback\b",                                               1.0),
    (r"\bk[i1]ke\b",                                               1.0),
    (r"\bch[i1]nks\b",                                             1.0),
    (r"\bgook\b",                                                   1.0),
    (r"\bcoon\b",                                                   0.9),
    (r"\bjigaboo\b",                                               1.0),
    (r"\bporch\s+monkey\b",                                        1.0),
    (r"\bsand\s+n[i1]gg[ae]r\b",                                   1.0),
    (r"\brag\s*head\b",                                            1.0),
    (r"\btowel\s*head\b",                                          1.0),
    (r"\bwh[i1]te\s+trash\b",                                      0.8),
    (r"\bcracker\b",                                               0.7),
    (r"\bh[a4]jj[i1]\b",                                           0.8),

    # Hate phrases
    (r"\bgo\s+back\s+to\s+(your\s+country|africa|mexico|china)\b", 1.0),
    (r"\byou\s+people\s+(are\s+all|should|don.t)\b",               0.7),
    (r"\ball\s+(blacks?|whites?|asians?|muslims?|jews?)\s+(are|should)\b", 0.8),
    (r"\bwhite\s+(power|supremacy|pride)\b",                       1.0),
    (r"\bblack\s+lives\s+(don.?t|do\s+not)\s+matter\b",           1.0),
    (r"\bgreat\s+replacement\b",                                   1.0),
    (r"\bethnic\s+cleansing\b",                                    1.0),
    (r"\bsubhuman\b",                                              0.9),
    (r"\bvermin\b",                                                0.8),
    (r"\bparasites?\b",                                            0.7),
]

ABUSIVE_PATTERNS = [
    # Threats
    (r"\bi.?ll\s+(kill|hurt|harm|destroy|ruin|end)\s+(you|ur|your)\b", 1.0),
    (r"\byou.?re\s+(dead|finished|done)\b",                        1.0),
    (r"\bwatch\s+your\s+back\b",                                   0.9),
    (r"\bi\s+know\s+where\s+you\s+live\b",                         1.0),
    (r"\bi\s+will\s+find\s+you\b",                                 0.9),
    (r"\bkill\s+yourself\b",                                       1.0),
    (r"\bkys\b",                                                   1.0),
    (r"\bgo\s+die\b",                                              1.0),
    (r"\bi.?ll\s+make\s+you\s+(pay|regret|suffer)\b",             1.0),

    # Harassment
    (r"\byou.?re\s+(worthless|pathetic|disgusting|ugly|stupid|idiot|moron)\b", 0.8),
    (r"\bnobody\s+(likes?|wants?|loves?)\s+you\b",                 0.8),
    (r"\byou\s+(deserve|should)\s+to\s+(die|suffer|hurt)\b",       1.0),
    (r"\bshut\s+(the\s+f[u\*]ck\s+)?up\b",                        0.7),
    (r"\bgo\s+f[u\*]ck\s+yourself\b",                              0.8),
    (r"\bpiece\s+of\s+sh[i\*]t\b",                                 0.8),
    (r"\byou\s+(f[u\*]cking\s+)?(loser|freak|pervert|creep)\b",    0.8),
    (r"\bi\s+hate\s+you\b",                                        0.7),
    (r"\byou\s+make\s+me\s+sick\b",                                0.7),

    # Bullying
    (r"\beveryone\s+(hates?|laughs?\s+at)\s+you\b",               0.9),
    (r"\byou\s+have\s+no\s+friends?\b",                           0.8),
    (r"\bno\s+one\s+(cares?|wants?\s+you)\b",                      0.8),
    (r"\byou.?re\s+a\s+(loser|failure|nothing)\b",                 0.8),
]


# ══════════════════════════════════════════════════════════════════════════════
#  SCORER
# ══════════════════════════════════════════════════════════════════════════════
def score_transcript(text: str) -> dict:
    """
    Score a transcript against all three categories.
    Returns dict: { 'sexual': float, 'racial': float, 'abusive': float }
    Scores are 0.0 – 1.0.
    """
    text_lower = text.lower()
    scores     = {"sexual": 0.0, "racial": 0.0, "abusive": 0.0}
    matched    = {"sexual": [], "racial": [], "abusive": []}

    categories = {
        "sexual":  SEXUAL_PATTERNS,
        "racial":  RACIAL_PATTERNS,
        "abusive": ABUSIVE_PATTERNS,
    }

    for category, patterns in categories.items():
        total_weight = 0.0
        for pattern, weight in patterns:
            hits = re.findall(pattern, text_lower, re.IGNORECASE)
            if hits:
                total_weight += weight * len(hits)
                matched[category].extend(hits)

        # Normalise: cap at 1.0, scale so 1 strong hit = ~0.7
        scores[category] = min(total_weight * 0.7, 1.0)

    return scores, matched # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL COLOURS
# ══════════════════════════════════════════════════════════════════════════════
class C:
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def red(s):    return f"{C.BOLD}{C.RED}{s}{C.RESET}"
def yellow(s): return f"{C.BOLD}{C.YELLOW}{s}{C.RESET}"
def green(s):  return f"{C.BOLD}{C.GREEN}{s}{C.RESET}"
def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def grey(s):   return f"{C.GREY}{s}{C.RESET}"


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════
def log_flag(category: str, score: float, transcript: str, matches: list):
    """Append a flag event to the local log file. No audio content stored."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{ts}] CATEGORY={category.upper()}  SCORE={score:.3f}  "
        f"MATCHES={matches}  TRANSCRIPT_SNIPPET={transcript[:80]!r}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT PRINTER
# ══════════════════════════════════════════════════════════════════════════════
def print_alert(category: str, score: float, transcript: str, matches: list):
    bar = "█" * int(score * 30)
    ts  = datetime.now().strftime("%H:%M:%S")

    category_colours = {
        "sexual":  red,
        "racial":  red,
        "abusive": yellow,
    }
    colour = category_colours.get(category, yellow)

    print()
    print(colour("━" * 60))
    print(colour(f"  ⚠  {category.upper()} CONTENT DETECTED  [{ts}]"))
    print(colour("━" * 60))
    print(f"  Score     : {colour(f'{score:.0%}')}  {colour(bar)}")
    print(f"  Matches   : {', '.join(str(m) for m in matches[:5])}")
    print(f"  Transcript: {cyan(repr(transcript[:100]))}")
    print(colour("━" * 60))
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO CAPTURE THREAD
# ══════════════════════════════════════════════════════════════════════════════
audio_queue: queue.Queue = queue.Queue()
running = True


def audio_capture_thread():
    """Captures microphone in chunks and puts numpy arrays on the queue."""
    try:
        import sounddevice as sd # type: ignore
    except ImportError:
        print("[ERROR] sounddevice not installed. Run: pip install sounddevice")
        return

    chunk_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
    print(green(f"[MIC]  Listening — capturing {CHUNK_SECONDS}s chunks at {SAMPLE_RATE}Hz"))

    while running:
        try:
            audio = sd.rec(
                chunk_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocking=True,
            )
            audio_queue.put(audio.squeeze())
        except Exception as e:
            print(f"[MIC ERROR] {e}")
            time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — TRANSCRIPTION + CLASSIFICATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global running

    print()
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}  AudioGuard — Real-Time Speech Threat Detector{C.RESET}")
    print(f"{C.BOLD}  Part of the SextortionGuard Project{C.RESET}")
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")
    print()
    print(f"  Model       : Whisper {WHISPER_MODEL} (runs fully locally)")
    print(f"  Threshold   : {CONF_THRESHOLD:.0%}")
    print(f"  Chunk size  : {CHUNK_SECONDS}s")
    print(f"  Log file    : {LOG_FILE}")
    print(f"  Categories  : SEXUAL  |  RACIAL  |  ABUSIVE")
    print()
    print(grey("  No audio is stored. No data leaves this machine."))
    print(grey("  Press Ctrl+C to quit cleanly."))
    print()

    # Load Whisper
    try:
        from faster_whisper import WhisperModel # type: ignore
    except ImportError:
        print("[ERROR] faster-whisper not installed.")
        print("        Run: pip install faster-whisper")
        sys.exit(1)

    print(f"[INFO] Loading Whisper '{WHISPER_MODEL}' model...")
    print(f"       First run downloads the model automatically.")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print(green("[INFO] Whisper ready.\n"))

    # Start mic capture thread
    t = threading.Thread(target=audio_capture_thread, daemon=True)
    t.start()

    chunk_count   = 0
    flag_count    = defaultdict(int)

    try:
        while True:
            # Wait for next audio chunk
            try:
                audio_chunk = audio_queue.get(timeout=2)
            except queue.Empty:
                continue

            chunk_count += 1

            # ── Transcribe ─────────────────────────────────────────────────
            try:
                segments, info = whisper.transcribe(
                    audio_chunk,
                    language=LANGUAGE,
                    beam_size=1,
                    vad_filter=True,          # skip silent chunks
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                    ),
                )
                transcript = " ".join(seg.text for seg in segments).strip()
            except Exception as e:
                print(f"[TRANSCRIBE ERROR] {e}")
                continue

            if not transcript:
                print(grey(f"  [{chunk_count:04d}] (silence)"))
                continue

            # ── Show live transcript ───────────────────────────────────────
            if SHOW_LIVE_TRANSCRIPT:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] {cyan(transcript)}")

            # ── Score transcript ───────────────────────────────────────────
            scores, matched = score_transcript(transcript)

            # ── Check thresholds ───────────────────────────────────────────
            triggered = False
            for category in ["sexual", "racial", "abusive"]:
                score = scores[category]
                if score >= CONF_THRESHOLD:
                    triggered = True
                    flag_count[category] += 1
                    print_alert(category, score, transcript, matched[category])
                    log_flag(category, score, transcript, matched[category])

            if not triggered:
                # Show a subtle score line when close to threshold
                max_score = max(scores.values())
                if max_score >= 0.30:
                    max_cat = max(scores, key=scores.get)
                    print(grey(
                        f"           ↑ low signal — {max_cat}: {max_score:.0%}"
                    ))

    except KeyboardInterrupt:
        running = False
        print()
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}  Session Summary{C.RESET}")
        print(f"{'═' * 60}")
        print(f"  Chunks processed : {chunk_count}")
        print(f"  Sexual flags     : {flag_count['sexual']}")
        print(f"  Racial flags     : {flag_count['racial']}")
        print(f"  Abusive flags    : {flag_count['abusive']}")
        print(f"  Log saved to     : {LOG_FILE}")
        print(f"{'═' * 60}")
        print(green("  Exited cleanly. No audio was stored."))
        print()


if __name__ == "__main__":
    main()