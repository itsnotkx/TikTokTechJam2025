# python pii_sanitizer_video.py -i myvideo.mp4 -o blurred.mp4 (command to execute)

import cv2
import re
import argparse
import numpy as np
import easyocr

# ---------- PII regex patterns ----------
PII_PATTERNS = {
    "phone":  re.compile(r"(?<!\d)(?:\+65\s?)?(?:\d{4}[\s-]?\d{4})(?!\d)"),
    "email":  re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "nric":   re.compile(r"\b[STFG]\d{7}[A-Z]\b", re.IGNORECASE),
    "postal": re.compile(r"\b\d{6}\b"),
    "url":    re.compile(r"(https?://[^\s]+|www\.[^\s]+)")
}

def is_pii(text: str) -> bool:
    return any(pattern.search(text.strip()) for pattern in PII_PATTERNS.values())

def pad_bbox(bbox, pad=0.1):
    """Expand OCR box by padding ratio."""
    pts = np.array(bbox, dtype=np.float32)
    x_min, y_min = pts[:,0].min(), pts[:,1].min()
    x_max, y_max = pts[:,0].max(), pts[:,1].max()
    w, h = x_max - x_min, y_max - y_min
    x_min -= w * pad; y_min -= h * pad
    x_max += w * pad; y_max += h * pad
    return int(x_min), int(y_min), int(x_max), int(y_max)

def blur_rect(frame, x1, y1, x2, y2):
    """Blur a rectangular region in the frame."""
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return
    roi_blur = cv2.GaussianBlur(roi, (51, 51), 30)
    frame[y1:y2, x1:x2] = roi_blur

def process_video(input_path, output_path, fps_sample=10, use_gpu=False):
    # Load OCR model
    reader = easyocr.Reader(['en'], gpu=use_gpu)

    cap = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    step = max(1, int(round(fps / fps_sample)))  # analyze ~10 fps

    print(f"[INFO] Processing {input_path} ({fps:.1f} fps) → {output_path}")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            results = reader.readtext(frame)
            for (bbox, text, conf) in results:
                if is_pii(text):
                    x1, y1, x2, y2 = pad_bbox(bbox, pad=0.1)
                    blur_rect(frame, x1, y1, x2, y2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] Saved to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to input video")
    ap.add_argument("-o", "--output", default="sanitized.mp4", help="Output path")
    ap.add_argument("--gpu", action="store_true", help="Use GPU if available")
    ap.add_argument("--fps", type=int, default=10, help="OCR sampling fps")
    args = ap.parse_args()

    process_video(args.input, args.output, fps_sample=args.fps, use_gpu=args.gpu)