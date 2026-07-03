#!/usr/bin/env python3
"""Concurrent load test for POST /deidentify-image.

Sends N requests with a pool of C concurrent workers and reports throughput
and latency percentiles.

The endpoint expects RAW pixel data plus dimensions (rows, columns,
bits_allocated, samples_per_pixel), not an encoded PNG/JPEG. This script
generates a synthetic grayscale frame with text, or reads --image (any format
OpenCV can decode) and sends it as raw grayscale.

Run inside the project's environment (requests + opencv are available there):
    poetry run python scripts/load_test.py -n 100 -c 20
    poetry run python scripts/load_test.py --image path/to/frame.png -n 50 -c 10
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests


def build_frame(image_path: str | None) -> tuple[bytes, int, int]:
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read image: {image_path}")
    else:
        img = np.full((200, 700), 255, np.uint8)
        cv2.putText(
            img,
            "PATIENT DUPONT 12345",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,),
            2,
        )
    h, w = img.shape
    return img.tobytes(), w, h


def one_request(url: str, raw: bytes, meta: dict) -> tuple[int, float]:
    files = {"image": ("frame.raw", raw, "application/octet-stream")}
    t0 = time.perf_counter()
    r = requests.post(url, files=files, data=meta, timeout=300)
    return r.status_code, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000/deidentify-image")
    ap.add_argument("-n", "--requests", type=int, default=50)
    ap.add_argument("-c", "--concurrency", type=int, default=10)
    ap.add_argument("--image", help="image file to send (decoded to grayscale)")
    args = ap.parse_args()

    raw, w, h = build_frame(args.image)
    meta = {
        "sensitive_data_list": json.dumps(
            {"PatientName": "DUPONT", "PatientID": "12345"}
        ),
        "rows": str(h),
        "columns": str(w),
        "bits_allocated": "8",
        "samples_per_pixel": "1",
    }

    latencies: list[float] = []
    codes: dict[object, int] = {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(one_request, args.url, raw, meta) for _ in range(args.requests)
        ]
        for fut in as_completed(futures):
            try:
                code, dt = fut.result()
            except Exception as exc:
                codes[f"error:{type(exc).__name__}"] = (
                    codes.get(f"error:{type(exc).__name__}", 0) + 1
                )
                continue
            codes[code] = codes.get(code, 0) + 1
            latencies.append(dt)
    total = time.perf_counter() - start

    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        return latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))]

    print(f"requests={args.requests} concurrency={args.concurrency} frame={w}x{h}")
    print(f"status codes: {codes}")
    print(f"total: {total:.2f}s  throughput: {args.requests / total:.2f} req/s")
    if latencies:
        print(
            f"latency s  min={latencies[0]:.3f} p50={pct(50):.3f} "
            f"p95={pct(95):.3f} max={latencies[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
