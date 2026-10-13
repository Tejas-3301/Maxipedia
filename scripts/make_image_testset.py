"""Generate the still-image ground-truth test set into data/image_testset/ (+ expected.json).

Base = a real photo (default data/images_test/f1_car.jpg) downscaled to 1600x900, which erases the source's JPEG
grid so every derived file starts with a single, known compression history.

  clean.jpg          one compression, camera-like custom quantisation tables, consistent EXIF + matching thumbnail
  spliced.jpg        300x300 region from another photo (pre-compressed q55 on a shifted grid, softened) pasted in;
                     EXIF incl. the ORIGINAL thumbnail copied from clean.jpg (what a careless tool does)
  resaved_3gen.jpg   clean re-saved three times with standard IJG tables (q80 -> q65 -> q50), EXIF lost
  exif_edited.jpg    clean's pixels untouched; ModifyDate moved after DateTimeOriginal, Photoshop Software tag,
                     embedded thumbnail replaced by another photo's
  screenshot.png     synthetic UI screenshot (lossless PNG, no JPEG history, no EXIF)

Needs exiftool on PATH (EXIF writing) and Pillow. expected.json is evaluation ground truth only.
Usage: python scripts/make_image_testset.py [--base data/images_test/f1_car.jpg] [--donor data/images_test/podium_aws.jpg]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W, H = 1600, 900
STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
STD_CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
              47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32
SHOT = "2025:03:02 14:05:11"


def camera_tables(seed: int = 7) -> list[list[int]]:
    """Non-IJG tables (IJG q92 perturbed by +-1), like a camera's own encoder."""
    rng = np.random.default_rng(seed)
    out = []
    for base in (STD_LUMA, STD_CHROMA):
        t = np.clip(np.floor((np.array(base) * 16 + 50) / 100), 1, 255)  # IJG scale for q92 = 16%
        t = np.clip(t + rng.integers(-1, 2, 64) * (t > 2), 1, 255)
        out.append([int(v) for v in t])
    return out


def jpeg_bytes(rgb: np.ndarray, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", **kw)
    return buf.getvalue()


def decode(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def exiftool(*args: str) -> None:
    subprocess.run(["exiftool", "-q", "-overwrite_original", *args], check=True, capture_output=True)


def thumb(rgb: np.ndarray, path: Path) -> Path:
    path.write_bytes(jpeg_bytes(cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA), quality=85))
    return path


def screenshot() -> np.ndarray:
    img = np.full((900, 1440, 3), 246, np.uint8)
    cv2.rectangle(img, (0, 0), (1440, 56), (32, 41, 64), -1)
    cv2.putText(img, "Race Control - Live Timing", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for i in range(12):
        y = 90 + i * 52
        cv2.rectangle(img, (24, y), (900, y + 44), (255, 255, 255) if i % 2 else (236, 240, 245), -1)
        cv2.putText(img, f"P{i + 1:<3} CAR {i * 7 + 1:>2}   1:{31 + i // 3:02d}.{(i * 137) % 1000:03d}   +{i * 0.412:.3f}",
                    (40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    cv2.rectangle(img, (940, 90), (1416, 520), (220, 226, 235), -1)
    cv2.circle(img, (1178, 305), 150, (66, 133, 244), 18)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "images_test" / "f1_car.jpg")
    ap.add_argument("--donor", type=Path, default=ROOT / "data" / "images_test" / "podium_aws.jpg")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "image_testset")
    args = ap.parse_args()
    if shutil.which("exiftool") is None:
        raise SystemExit("exiftool not found on PATH")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)

    base = cv2.cvtColor(cv2.resize(cv2.imread(str(args.base)), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    cam_q = camera_tables()
    exif_args = [f"-Make=Canon", "-Model=Canon EOS R6", f"-DateTimeOriginal={SHOT}", f"-CreateDate={SHOT}",
                 f"-ModifyDate={SHOT}", "-Orientation#=1", "-ExposureTime=1/1000", "-FNumber=5.6", "-ISO=400"]

    # clean
    p_clean = out / "clean.jpg"
    p_clean.write_bytes(jpeg_bytes(base, qtables=cam_q, subsampling=2))
    clean_px = decode(p_clean.read_bytes())
    exiftool(*exif_args, f"-ThumbnailImage<={thumb(clean_px, work / 'thumb_clean.jpg')}", str(p_clean))

    # spliced: donor region, pre-compressed q55 on a shifted grid, softened, pasted at a non-multiple-of-8 offset
    donor = cv2.cvtColor(cv2.imread(str(args.donor)), cv2.COLOR_BGR2RGB)
    dh, dw = donor.shape[:2]
    patch = cv2.resize(donor[dh // 6: dh // 6 + dh // 2, dw // 3: dw // 3 + dh // 2], (300, 300), interpolation=cv2.INTER_AREA)
    patch = decode(jpeg_bytes(patch, quality=55))
    patch = cv2.GaussianBlur(patch, (0, 0), 0.8)
    x, y = 1043, 211
    spliced_px = base.copy()
    spliced_px[y:y + 300, x:x + 300] = patch
    p_spl = out / "spliced.jpg"
    p_spl.write_bytes(jpeg_bytes(spliced_px, qtables=cam_q, subsampling=2))
    exiftool("-TagsFromFile", str(p_clean), "-all:all", str(p_spl))  # incl. the original thumbnail

    # three re-save generations with standard IJG tables, EXIF dropped
    b = p_clean.read_bytes()
    for q in (80, 65, 50):
        b = jpeg_bytes(decode(b), quality=q)
    (out / "resaved_3gen.jpg").write_bytes(b)

    # EXIF edited, pixels untouched
    p_ex = out / "exif_edited.jpg"
    shutil.copy(p_clean, p_ex)
    other = cv2.cvtColor(cv2.resize(donor, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    exiftool("-ModifyDate=2026:09:30 22:41:07", "-Software=Adobe Photoshop 25.1 (Windows)",
             f"-ThumbnailImage<={thumb(other, work / 'thumb_other.jpg')}", str(p_ex))

    # PNG screenshot
    cv2.imwrite(str(out / "screenshot.png"), screenshot())
    shutil.rmtree(work)

    sha = lambda p: hashlib.sha256((out / p).read_bytes()).hexdigest()
    expected = {
        "_note": "Evaluation ground truth only. Checks and fusion must never read this file.",
        "base": str(args.base.name), "donor": str(args.donor.name),
        "files": {
            "clean.jpg": {"ground_truth": "authentic", "acceptable_verdicts": ["likely authentic", "inconclusive"],
                          "notes": "single compression, camera-like tables, consistent EXIF, matching thumbnail"},
            "spliced.jpg": {"ground_truth": "manipulated", "acceptable_verdicts": ["likely manipulated", "inconclusive"],
                            "region": [x, y, 300, 300],
                            "notes": "pasted region with its own compression history; original thumbnail kept"},
            "resaved_3gen.jpg": {"ground_truth": "authentic_content_recompressed",
                                 "acceptable_verdicts": ["inconclusive", "likely authentic"],
                                 "notes": "IJG q80 -> q65 -> q50, EXIF lost"},
            "exif_edited.jpg": {"ground_truth": "manipulated_metadata",
                                "acceptable_verdicts": ["likely manipulated", "inconclusive"],
                                "notes": "ModifyDate > DateTimeOriginal, Photoshop Software, foreign thumbnail"},
            "screenshot.png": {"ground_truth": "authentic_screenshot", "acceptable_verdicts": ["inconclusive", "likely authentic"],
                               "notes": "lossless PNG, no JPEG history or EXIF: most image checks not assessable"},
        },
    }
    for name, e in expected["files"].items():
        e["sha256"] = sha(name)
    (out / "expected.json").write_text(json.dumps(expected, indent=2))
    print("\n".join(f"{n}: {(out / n).stat().st_size} bytes" for n in expected["files"]))


if __name__ == "__main__":
    main()
