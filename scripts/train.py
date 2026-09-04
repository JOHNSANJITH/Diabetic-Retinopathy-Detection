import argparse
from pathlib import Path
import tensorflow as tf
from drd.config import load_config
from drd.data import make_datasets, class_weights
from drd.model import build_model

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.json"); args = parser.parse_args()
    cfg = load_config(args.config); tf.keras.utils.set_random_seed(cfg.seed)
    train, validation, _ = make_datasets(cfg.data_dir, cfg.image_size, cfg.batch_size, cfg.seed)
    model = build_model(cfg.image_size, cfg.dropout, cfg.learning_rate, cfg.fine_tune, cfg.fine_tune_layers)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(cfg.model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    model.fit(train, validation_data=validation, epochs=cfg.epochs, class_weight=class_weights(cfg.data_dir / "train"), callbacks=callbacks)
    print(f"Saved best model to {cfg.model_path}")

if __name__ == "__main__": main()
