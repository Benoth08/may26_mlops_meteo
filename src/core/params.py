from pathlib import Path
import yaml

PARAMS_FILE = Path(__file__).parent.parent / "params.yml"

def load_params():
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)