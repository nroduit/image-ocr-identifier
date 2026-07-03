import os

from dotenv import dotenv_values
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Read OCR_MODEL from .env to bundle only the selected model.
_env = dotenv_values(os.path.join(SPECPATH, ".env"))
_ocr_model = _env.get("OCR_MODEL", "PP-OCRv5_mobile")
_onnx_base = os.path.join("data", "models", "onnx")

# Resolve which folders to bundle based on the model name.
_single_dir = os.path.join(_onnx_base, _ocr_model)
if os.path.isdir(_single_dir):
    # v5 style: single folder
    _model_dirs = [(_single_dir, os.path.join(_onnx_base, _ocr_model))]
else:
    # v6 style: separate det/rec folders
    _det = f"{_ocr_model}_det_onnx"
    _rec = f"{_ocr_model}_rec_onnx"
    _model_dirs = [
        (os.path.join(_onnx_base, _det), os.path.join(_onnx_base, _det)),
        (os.path.join(_onnx_base, _rec), os.path.join(_onnx_base, _rec)),
    ]

datas = []
binaries = []
hiddenimports = [
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "image_ocr_identifier.main",
    "image_ocr_identifier.routers.deidentify_image",
    "image_ocr_identifier.routers.reporting",
    "image_ocr_identifier.ocr._rapid",
]
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("image_ocr_identifier")

for pkg in ("rapidocr_onnxruntime", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h


def _exclude_conflicting_openssl_libs(entries):
    blocked = {"libcrypto.3.dylib", "libssl.3.dylib"}
    filtered = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) < 2:
            filtered.append(entry)
            continue

        src = entry[0]
        dest = entry[1]
        src_name = os.path.basename(str(src))
        dest_name = os.path.basename(str(dest))

        if src_name in blocked or dest_name in blocked:
            continue

        filtered.append(entry)

    return filtered


binaries = _exclude_conflicting_openssl_libs(binaries)

# cv2 may bundle an older libcrypto/libssl that shadows the one Python's _ssl
# needs. Inject Homebrew's OpenSSL into cv2/.dylibs when present (macOS build).
openssl_dir = "/opt/homebrew/opt/openssl@3/lib"
for _lib in ("libcrypto.3.dylib", "libssl.3.dylib"):
    _src = os.path.join(openssl_dir, _lib)
    if os.path.exists(_src):
        binaries.append((_src, "cv2/.dylibs"))

datas += [
    # ONNX OCR models selected by OCR_MODEL in .env.
    *_model_dirs,
]

a = Analysis(
    ["server_portable.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "IPython",
        "notebook",
        "PyQt5",
        "PySide2",
        "PIL.ImageQt",
        # OCR runs on onnxruntime via RapidOCR in the portable build. Exclude
        # the heavy DL engines (still present in the dev venv) so they are not
        # bundled (~630 MB saved).
        "paddle",
        "paddleocr",
        "paddlex",
        "torch",
        "torchvision",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="image-ocr-identifier",
    console=True,
    strip=True,  # Linux/macOS : retire les symboles
    upx=True,  # nécessite UPX installé
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name="image-ocr-identifier",
)
