"""
test_pitchtime_api.py — End-to-end test of the 变调与变速 REST endpoints.

Run:  python3 tests/test_pitchtime_api.py
"""

import base64
import io
import json
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import app as webapp  # noqa: E402
except ModuleNotFoundError:  # Flask not installed in this environment
    print("SKIP: Flask is not installed — API end-to-end test skipped.")
    sys.exit(0)
from backend import audio_io, dsp  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def main():
    webapp.app.config["TESTING"] = True
    c = webapp.app.test_client()

    # Page renders and is registered in the nav.
    r = c.get("/pitchtempo")
    check("page /pitchtempo renders", r.status_code == 200)
    check("page mentions WSOLA", "WSOLA" in r.get_data(as_text=True))

    # Generate a 3 s 440 Hz sine through the library API.
    r = c.post("/api/library/generate",
               data=json.dumps({"kind": "sine", "sr": 44100, "duration": 3.0, "freq": 440}),
               content_type="application/json")
    check("generate sine", r.status_code == 200)
    src = r.get_json()
    src_id = src["id"]
    check("source duration 3 s", abs(src["duration"] - 3.0) < 0.01,
          f"{src['duration']}")

    # --- preview endpoint ---------------------------------------------------
    r = c.post("/api/pitchtempo/preview",
               data=json.dumps({"file_id": src_id, "semitones": 4, "tempo": 1.5,
                                "start": 0, "duration": 4}),
               content_type="application/json")
    check("preview 200", r.status_code == 200)
    payload = r.get_json()
    raw = base64.b64decode(payload["wav"])
    w = wave.open(io.BytesIO(raw))
    prev_dur = w.getnframes() / w.getframerate()
    check("preview duration = 4 s / 1.5", abs(prev_dur - 4 / 1.5) < 0.05,
          f"{prev_dur:.3f}s")

    # --- apply: tempo only ---------------------------------------------------
    r = c.post("/api/pitchtempo/apply",
               data=json.dumps({"file_id": src_id, "semitones": 0, "tempo": 1.5}),
               content_type="application/json")
    check("apply tempo 200", r.status_code == 200)
    e1 = r.get_json()
    check("tempo-only duration", abs(e1["duration"] - 2.0) < 0.01,
          f"{e1['duration']:.4f}s")
    check("registered as derived", e1.get("derived_from") == src_id)
    check("params stored", e1.get("pitchtempo", {}).get("tempo") == 1.5)
    with audio_io.WavReader(os.path.join(webapp.DATA_DIR, e1["path"])) as rd:
        chunk = rd.read_chunk(1 << 20)
    f0 = dsp.pitch_autocorr(chunk[0][44100:44100 + 4096], 44100)
    check("tempo-only pitch unchanged", f0 and abs(f0 - 440) < 9,
          f"f0={f0 and round(f0, 1)}")

    # --- apply: pitch only ---------------------------------------------------
    r = c.post("/api/pitchtempo/apply",
               data=json.dumps({"file_id": src_id, "semitones": 7, "tempo": 1.0}),
               content_type="application/json")
    e2 = r.get_json()
    check("pitch-only duration", abs(e2["duration"] - 3.0) < 0.01,
          f"{e2['duration']:.4f}s")
    with audio_io.WavReader(os.path.join(webapp.DATA_DIR, e2["path"])) as rd:
        chunk = rd.read_chunk(1 << 20)
    f0 = dsp.pitch_autocorr(chunk[0][44100:44100 + 4096], 44100)
    want = 440 * 2 ** (7 / 12)
    check("pitch-only +7st", f0 and abs(f0 - want) / want < 0.02,
          f"f0={f0 and round(f0, 1)} want {want:.1f}")

    # --- apply: both, then it shows up in the library -----------------------
    r = c.post("/api/pitchtempo/apply",
               data=json.dumps({"file_id": src_id, "semitones": -3, "tempo": 0.8}),
               content_type="application/json")
    e3 = r.get_json()
    check("combined duration", abs(e3["duration"] - 3.0 / 0.8) < 0.01,
          f"{e3['duration']:.4f}s")
    r = c.get("/api/library")
    ids = [f["id"] for f in r.get_json()]
    check("all derived files in library",
          all(e["id"] in ids for e in (e1, e2, e3)))

    # --- validation ----------------------------------------------------------
    r = c.post("/api/pitchtempo/apply",
               data=json.dumps({"file_id": "nonexistent", "semitones": 1}),
               content_type="application/json")
    check("unknown file -> 404", r.status_code == 404)
    r = c.post("/api/pitchtempo/apply",
               data=json.dumps({"file_id": src_id, "semitones": 99, "tempo": 99}),
               content_type="application/json")
    e4 = r.get_json()
    check("params clamped", e4["pitchtempo"]["semitones"] == 24.0
          and e4["pitchtempo"]["tempo"] == 4.0,
          str(e4.get("pitchtempo")))

    print(f"\n{'=' * 50}\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
