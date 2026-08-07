import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = REPO_ROOT / "data" / "models" / "onnx"

MODELS = {
    "PP-OCRv5_mobile": {
        "det": "PaddlePaddle/PP-OCRv5_mobile_det_onnx",
        "rec": "PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx",
        "layout": "v5",
    },
    "PP-OCRv6_small": {
        "det": "PaddlePaddle/PP-OCRv6_small_det_onnx",
        "rec": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "layout": "v6",
    },
    "PP-OCRv6_medium": {
        "det": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        "rec": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        "layout": "v6",
    },
    "PP-OCRv6_tiny": {
        "det": "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        "rec": "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        "layout": "v6",
    },
}


def _download(repo_id, filename, dest_dir, dest_name=None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id, filename)
    target = dest_dir / (dest_name or filename)
    shutil.copyfile(cached, target)
    return target


def _write_keys(yml_path, keys_path):
    data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    chars = data["PostProcess"]["character_dict"]
    keys_path.write_text("\n".join(chars) + "\n", encoding="utf-8")


def fetch(model_name):
    cfg = MODELS[model_name]
    if cfg["layout"] == "v5":
        dest = ONNX_DIR / model_name
        _download(cfg["det"], "inference.onnx", dest, "det.onnx")
        _download(cfg["rec"], "inference.onnx", dest, "rec.onnx")
        yml = _download(cfg["rec"], "inference.yml", dest, "rec.yml")
        _write_keys(yml, dest / "keys.txt")
        yml.unlink()
    else:
        det_dir = ONNX_DIR / f"{model_name}_det_onnx"
        rec_dir = ONNX_DIR / f"{model_name}_rec_onnx"
        _download(cfg["det"], "inference.onnx", det_dir)
        _download(cfg["rec"], "inference.onnx", rec_dir)
        yml = _download(cfg["rec"], "inference.yml", rec_dir)
        _write_keys(yml, rec_dir / "keys.txt")


if __name__ == "__main__":
    default = dotenv_values(REPO_ROOT / ".env").get("OCR_MODEL", "PP-OCRv5_mobile")
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OCR_MODEL", default)
    if model not in MODELS:
        sys.exit(f"OCR_MODEL inconnu: {model}. Choix: {list(MODELS)}")
    fetch(model)
    print(f"Modèle ONNX '{model}' récupéré dans {ONNX_DIR}")
