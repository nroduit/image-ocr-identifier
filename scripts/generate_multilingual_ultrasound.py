"""Generate a fake ultrasound image with multilingual burned-in text.

Creates a synthetic ultrasound-style image with patient metadata in
multiple scripts (Latin, Japanese, Arabic, Hebrew, Russian, Korean, Chinese)
and outputs a matching sensitive_data_list JSON for API testing.

Usage:
    python scripts/generate_multilingual_ultrasound.py

Outputs:
    data/multilingual_ultrasound.png
    data/multilingual_sensitive_data.json
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# Simulated patient metadata per script
PATIENT_RECORDS = {
    "latin": {
        "name": "Dupont^Marie",
        "id": "PAT-20231045",
        "birthdate": "15/03/1987",
        "sex": "F",
        "study_date": "30/06/2026",
    },
    "japanese_hiragana": {
        "name": "やまだ たろう",
        "label": "患者名",
    },
    "japanese_katakana": {
        "name": "ヤマダ タロウ",
        "label": "カタカナ",
    },
    "arabic": {
        "name": "محمد أحمد",
        "label": "اسم المريض",
    },
    "hebrew": {
        "name": "יוסף כהן",
        "label": "שם המטופל",
    },
    "russian": {
        "name": "Иванов Сергей",
        "label": "Пациент",
    },
    "korean": {
        "name": "김민수",
        "label": "환자명",
    },
    "chinese": {
        "name": "王小明",
        "label": "患者姓名",
    },
}


def find_font(preferred: list[str], size: int) -> ImageFont.FreeTypeFont:
    """Try preferred fonts, fall back to a system font that covers many scripts."""
    # macOS paths
    font_dirs = [
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
    ]
    for name in preferred:
        for d in font_dirs:
            for ext in (".ttf", ".ttc", ".otf"):
                p = d / f"{name}{ext}"
                if p.exists():
                    return ImageFont.truetype(str(p), size)
    # Absolute fallback
    return ImageFont.load_default()


def draw_ultrasound_background(img: np.ndarray) -> np.ndarray:
    """Draw a simple fake ultrasound cone pattern on a black background."""
    h, w = img.shape[:2]
    cx = w // 2
    for y in range(h):
        spread = int(0.6 * y)
        x_start = max(cx - spread, 0)
        x_end = min(cx + spread, w)
        if x_start < x_end:
            noise = np.random.randint(5, 45, size=(x_end - x_start,), dtype=np.uint8)
            # Darken toward edges with a gradient
            grad = np.linspace(0.3, 1.0, (x_end - x_start) // 2 + 1)
            grad = np.concatenate([grad, grad[::-1]])[: x_end - x_start]
            img[y, x_start:x_end] = (noise * grad).astype(np.uint8)
    return img


def generate_image():
    width, height = 1024, 768
    img = np.zeros((height, width), dtype=np.uint8)
    img = draw_ultrasound_background(img)

    # Convert to RGB PIL image
    pil_img = Image.fromarray(np.stack([img] * 3, axis=-1))
    draw = ImageDraw.Draw(pil_img)

    # Font selection - try Noto/Arial Unicode for broad coverage
    font_large = find_font(
        [
            "Arial Unicode",
            "Arial Unicode MS",
            "NotoSansCJK-Regular",
            "Hiragino Sans W3",
        ],
        22,
    )
    font_small = find_font(
        [
            "Arial Unicode",
            "Arial Unicode MS",
            "NotoSansCJK-Regular",
            "Hiragino Sans W3",
        ],
        18,
    )
    font_latin = find_font(["Helvetica", "Arial", "DejaVuSans"], 20)

    text_color = (220, 220, 220)  # Light gray like real burned-in text
    accent_color = (180, 200, 220)  # Slightly blue tint

    # Top-left: Latin patient info
    latin = PATIENT_RECORDS["latin"]
    lines_top_left = [
        f"{latin['name'].replace('^', ' ')}",
        f"ID: {latin['id']}",
        f"DOB: {latin['birthdate']}",
        "Sex: F",
        f"Study: {latin['study_date']}",
    ]
    y_pos = 15
    for line in lines_top_left:
        draw.text((15, y_pos), line, fill=text_color, font=font_latin)
        y_pos += 26

    # Top-right: Japanese (hiragana + katakana)
    jp_hira = PATIENT_RECORDS["japanese_hiragana"]
    jp_kata = PATIENT_RECORDS["japanese_katakana"]
    right_x = width - 300
    draw.text(
        (right_x, 15),
        f"{jp_hira['label']}: {jp_hira['name']}",
        fill=text_color,
        font=font_large,
    )
    draw.text(
        (right_x, 45),
        f"{jp_kata['label']}: {jp_kata['name']}",
        fill=accent_color,
        font=font_large,
    )

    # Middle-left: Arabic
    ar = PATIENT_RECORDS["arabic"]
    draw.text(
        (15, height - 180),
        f"{ar['label']}: {ar['name']}",
        fill=text_color,
        font=font_large,
    )

    # Middle-right: Hebrew
    he = PATIENT_RECORDS["hebrew"]
    draw.text(
        (right_x, height - 180),
        f"{he['label']}: {he['name']}",
        fill=text_color,
        font=font_large,
    )

    # Bottom-left: Russian
    ru = PATIENT_RECORDS["russian"]
    draw.text(
        (15, height - 130),
        f"{ru['label']}: {ru['name']}",
        fill=accent_color,
        font=font_large,
    )

    # Bottom-center: Korean
    ko = PATIENT_RECORDS["korean"]
    draw.text(
        (width // 2 - 80, height - 130),
        f"{ko['label']}: {ko['name']}",
        fill=text_color,
        font=font_large,
    )

    # Bottom-right: Chinese
    zh = PATIENT_RECORDS["chinese"]
    draw.text(
        (right_x, height - 130),
        f"{zh['label']}: {zh['name']}",
        fill=text_color,
        font=font_large,
    )

    # Bottom bar with additional info
    draw.rectangle([(0, height - 50), (width, height)], fill=(20, 20, 20))
    draw.text(
        (15, height - 40),
        "MI 0.7  TIS 0.4  FPS 28",
        fill=(150, 150, 150),
        font=font_small,
    )
    draw.text(
        (right_x, height - 40),
        "2D / B-Mode  L12-5",
        fill=(150, 150, 150),
        font=font_small,
    )

    # Depth markers on the right edge
    for i in range(1, 6):
        marker_y = int(height * i / 6)
        draw.text(
            (width - 45, marker_y), f"{i * 2}cm", fill=(100, 100, 100), font=font_small
        )

    return pil_img


def generate_sensitive_data() -> dict:
    """Build a sensitive_data_list dict as the API expects it."""
    return {
        # Latin
        "PatientName": "Dupont^Marie",
        "PatientID": "PAT-20231045",
        "PatientBirthDate": "19870315",
        "PatientSex": "F",
        "StudyDate": "20260630",
        # Japanese
        "PatientNameJP": "やまだ たろう",
        "PatientNameKata": "ヤマダ タロウ",
        # Arabic
        "PatientNameAR": "محمد أحمد",
        # Hebrew
        "PatientNameHE": "יוסף כהן",
        # Russian
        "PatientNameRU": "Иванов Сергей",
        # Korean
        "PatientNameKO": "김민수",
        # Chinese
        "PatientNameZH": "王小明",
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate image
    pil_img = generate_image()
    img_path = OUTPUT_DIR / "multilingual_ultrasound.png"
    pil_img.save(str(img_path))
    print(f"Image saved (PNG preview): {img_path}")

    # Save as raw pixel data (BGR, 8-bit) for the API
    img_array = np.array(pil_img)  # RGB
    img_bgr = img_array[:, :, ::-1].copy()  # Convert to BGR
    rows, columns = img_bgr.shape[:2]
    bits_allocated = 8
    samples_per_pixel = 3

    raw_path = OUTPUT_DIR / "multilingual_ultrasound.raw"
    raw_path.write_bytes(img_bgr.tobytes())
    print(f"Image saved (raw pixels): {raw_path}")

    # Generate sensitive data JSON
    sensitive_data = generate_sensitive_data()
    json_path = OUTPUT_DIR / "multilingual_sensitive_data.json"
    json_path.write_text(json.dumps(sensitive_data, ensure_ascii=False, indent=2))
    # Compact version for curl (no newlines)
    json_compact_path = OUTPUT_DIR / "multilingual_sensitive_data_compact.json"
    json_compact_path.write_text(json.dumps(sensitive_data, ensure_ascii=False))
    print(f"Sensitive data saved: {json_path}")

    # Print curl command for testing
    print("\n--- Test command ---")
    print(
        f"curl -X POST http://localhost:8000/deidentify-image \\\n"
        f'  -H "Accept: application/json; version=1" \\\n'
        f'  -F "image=@{raw_path};type=application/octet-stream" \\\n'
        f'  -F "sensitive_data_list=<{json_compact_path}" \\\n'
        f'  -F "rows={rows}" \\\n'
        f'  -F "columns={columns}" \\\n'
        f'  -F "bits_allocated={bits_allocated}" \\\n'
        f'  -F "samples_per_pixel={samples_per_pixel}"'
    )


if __name__ == "__main__":
    main()
