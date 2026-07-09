import multiprocessing
import os
import sys


def _setup_env() -> None:
    # The portable bundle ships the RapidOCR/ONNX backend.
    os.environ.setdefault("OCR_BACKEND", "rapid")

    # Modèles embarqués dans le bundle (voir le .spec). En onedir, _MEIPASS
    # pointe vers le dossier _internal/.
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
        os.environ.setdefault("MODEL_DATA_PATH", os.path.join(base, "data"))

        # Auto-detect bundled OCR model if OCR_MODEL is not already set.
        if "OCR_MODEL" not in os.environ:
            onnx_dir = os.path.join(base, "data", "models", "onnx")
            if os.path.isdir(onnx_dir):
                for entry in os.listdir(onnx_dir):
                    path = os.path.join(onnx_dir, entry)
                    if os.path.isdir(path) and not entry.startswith("."):
                        # v6 style: strip _det_onnx suffix
                        if entry.endswith("_det_onnx"):
                            os.environ["OCR_MODEL"] = entry.removesuffix("_det_onnx")
                            break
                        # v5 style: folder name is the model name directly
                        elif os.path.isfile(os.path.join(path, "det.onnx")):
                            os.environ["OCR_MODEL"] = entry
                            break

    # Portable : un seul worker, pas de multiprocessing fragile sous gel.
    os.environ.setdefault("WORKERS", "1")


def _log_dir() -> str:
    # Next to the executable when frozen, next to this script otherwise.
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _run() -> None:
    _setup_env()
    import uvicorn

    # Import the app eagerly so import errors are caught here (and logged)
    # instead of being swallowed by uvicorn's import machinery under PyInstaller.
    from image_ocr_identifier.main import app

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),  # local uniquement
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        h11_max_incomplete_event_size=50 * 1024 * 1024,
    )


def main() -> None:
    multiprocessing.freeze_support()  # indispensable si un worker est spawn
    try:
        _run()
    except Exception:
        import traceback

        tb = traceback.format_exc()
        log_path = os.path.join(_log_dir(), "deidentify-karnak-crash.log")
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(tb)
        except OSError:
            pass
        sys.stderr.write(tb)
        sys.stderr.flush()
        # Keep the console window open so the error is readable when the exe is
        # launched by double-click (Windows) instead of from a terminal.
        if getattr(sys, "frozen", False):
            try:
                input("\nA fatal error occurred. Press Enter to close...")
            except (EOFError, KeyboardInterrupt):
                pass
        raise


if __name__ == "__main__":
    main()
