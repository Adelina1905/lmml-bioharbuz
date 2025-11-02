import cv2
import numpy as np
import os
from pathlib import Path

IMG_PATH = r"e:\hackaton\distorted_qr.png"

# Try decoding with OpenCV QRCodeDetector
def try_decode(img):
    detector = cv2.QRCodeDetector()
    data, points, straight_qrcode = detector.detectAndDecode(img)
    if data:
        return data
    return None


def try_decode_pyzbar(img):
    try:
        from pyzbar import pyzbar
    except Exception:
        return None
    # pyzbar expects grayscale or BGR
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    bars = pyzbar.decode(gray)
    for b in bars:
        if b.data:
            return b.data.decode('utf-8', errors='replace')
    return None


def preprocess_variants(img):
    # yield variants of image for decoding attempts
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield ("orig", img)
    yield ("gray", gray)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    yield ("clahe", gray_clahe)

    # Denoise
    den = cv2.fastNlMeansDenoising(gray, None, h=10)
    yield ("denoise", den)

    # Bilateral filter
    bil = cv2.bilateralFilter(gray, 9, 75, 75)
    yield ("bilateral", bil)

    # Adaptive threshold
    at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 10)
    yield ("adaptiveth", at)

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield ("otsu", otsu)

    # Morphological close to fill
    kernel = np.ones((3,3), np.uint8)
    close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    yield ("close", close)

    # Median blur
    med = cv2.medianBlur(gray, 5)
    yield ("median", med)

    # Resize variations
    for scale in [1.0, 1.5, 2.0, 3.0]:
        h,w = gray.shape[:2]
        imgr = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
        yield (f"resize_{scale}", imgr)

    # rotate variations
    for angle in [0, 90, 180, 270]:
        M = cv2.getRotationMatrix2D((gray.shape[1]/2, gray.shape[0]/2), angle, 1)
        rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        yield (f"rot_{angle}", rot)


if __name__ == '__main__':
    if not os.path.exists(IMG_PATH):
        print(f"ERROR: file not found: {IMG_PATH}")
        exit(1)

    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"ERROR: OpenCV could not read image: {IMG_PATH}")
        exit(1)

    found = None
    # Try direct detection on color
    dd = try_decode(img)
    if dd:
        print("DECODED:", dd)
        found = dd

    # Try variants
    if not found:
        for name, var in preprocess_variants(img):
            # ensure 3-channel for detectAndDecode if needed
            if len(var.shape) == 2:
                test_img = var
            else:
                test_img = var
            try:
                res = try_decode(test_img)
            except Exception as e:
                res = None
            if res:
                print(f"DECODED (variant={name}): {res}")
                found = res
                break

    # Try pyzbar on original and variants if available
    if not found:
        try:
            from pyzbar import pyzbar
            pyz_available = True
        except Exception:
            pyz_available = False
        if pyz_available:
            res = try_decode_pyzbar(img)
            if res:
                print(f"DECODED (pyzbar=orig): {res}")
                found = res
        if not found:
            for name, var in preprocess_variants(img):
                try:
                    res = try_decode_pyzbar(var)
                except Exception:
                    res = None
                if res:
                    print(f"DECODED (pyzbar_variant={name}): {res}")
                    found = res
                    break

    # As a last resort try threshold+contour extraction to isolate largest square-like region
    if not found:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) >= 4:
                x,y,w,h = cv2.boundingRect(approx)
                candidates.append((area, x,y,w,h))
        candidates.sort(reverse=True)
        for i, (area,x,y,w,h) in enumerate(candidates[:5]):
            crop = img[y:y+h, x:x+w]
            res = try_decode(crop)
            if res:
                print(f"DECODED (crop#{i}): {res}")
                found = res
                break

    if not found:
        print("NO_DECODE")
    else:
        # Save a copy of the straight_qrcode if detector provided one for inspection
        out_dir = Path(r"e:\hackaton")
        try:
            detector = cv2.QRCodeDetector()
            _, _, sq = detector.detectAndDecode(img)
            if isinstance(sq, np.ndarray):
                cv2.imwrite(str(out_dir / "straight_qrcode_extracted.png"), sq)
        except Exception:
            pass
        # Finally print only the decoded string as requested by the challenge
        print(found)
