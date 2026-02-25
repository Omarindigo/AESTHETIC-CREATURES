Run in a venv:

Windows (PowerShell):
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt

macOS/Linux:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

1) Live PyBullet GUI (pick creature):
  python scripts/run_gui.py --creature starfish
  python scripts/run_gui.py --creature swimmer
  python scripts/run_gui.py --creature tumbler

2) Record one specimen to JSON (headless by default; add --gui to watch):
  python scripts/rollout_record.py --creature starfish --out specimen_starfish.json --seconds 12
  python scripts/rollout_record.py --creature swimmer --out specimen_swimmer.json --seconds 12

3) Generate a small dataset of specimens:
  python scripts/rollout_dataset.py --creature starfish --count 10 --out_dir specimens

4) Replay in Three.js viewer:
  Start a simple local server in the project root:
    python -m http.server 8000
  Open:
    http://localhost:8000/viewer/
  Then load your JSON file using the file picker.