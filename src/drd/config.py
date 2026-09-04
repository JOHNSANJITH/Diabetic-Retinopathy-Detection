from dataclasses import dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class Config:
    data_dir: Path
    model_path: Path
    image_size: tuple[int, int] = (299, 299)
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 1e-4
    validation_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42
    fine_tune: bool = False
    fine_tune_layers: int = 30
    dropout: float = 0.35

def load_config(path: str | Path) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["data_dir"] = Path(raw["data_dir"])
    raw["model_path"] = Path(raw["model_path"])
    raw["image_size"] = tuple(raw.get("image_size", (299, 299)))
    return Config(**raw)
