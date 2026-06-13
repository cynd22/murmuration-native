# Audio feeder

Captures system audio (PipeWire/PulseAudio), runs FFT band analysis, per-band
onset detection, and realtime beat tracking; pushes JSON frames to the
visualiser over a websocket.

```sh
cd feeder
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python feeder_onsets.py
```

Then launch the visualiser — it connects to ws://localhost:8766 automatically.
The visualiser also runs fine without the feeder (calm, music-less flock).
