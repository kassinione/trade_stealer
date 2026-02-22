#!/usr/bin/env python3
"""Screen price monitor.

Раз в секунду:
1) Ищет блок "Запросов на покупку ..." и вытаскивает цену справа -> p1
2) Ищет список с кнопками "КУПИТЬ" и вытаскивает самую верхнюю цену слева -> p2

Требует локально установленные:
- tesseract-ocr
- python: mss opencv-python pytesseract numpy
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mss
import numpy as np
import pytesseract

PRICE_RE = re.compile(r"(\d+[\.,]\d{2})\s*G", re.IGNORECASE)


@dataclass
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float


class PriceMonitor:
    def __init__(self, interval_sec: float = 1.0) -> None:
        self.interval_sec = interval_sec
        self.ocr_languages = "rus+eng"

    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
        return cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def yellow_text_mask(image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([12, 80, 80], dtype=np.uint8)
        upper = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 3)
        return cv2.resize(mask, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    def _ocr_data(self, image: np.ndarray) -> list[OcrWord]:
        cfg = "--oem 3 --psm 6"
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.ocr_languages,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError:
            data = pytesseract.image_to_data(
                image,
                lang="eng",
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )

        words: list[OcrWord] = []
        for i, txt in enumerate(data["text"]):
            txt = (txt or "").strip()
            if not txt:
                continue
            conf_str = data["conf"][i]
            try:
                conf = float(conf_str)
            except (TypeError, ValueError):
                conf = -1.0
            words.append(
                OcrWord(
                    text=txt,
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                    conf=conf,
                )
            )
        return words

    @staticmethod
    def parse_price(text: str) -> Optional[float]:
        m = PRICE_RE.search(text)
        if not m:
            return None
        return float(m.group(1).replace(",", "."))

    def find_p1(self, screen_bgr: np.ndarray) -> Optional[float]:
        prep = self.preprocess(screen_bgr)
        words = self._ocr_data(prep)

        # Ищем строку с "Запросов" + "покуп"
        anchor_y = None
        for w in words:
            t = w.text.lower()
            if "запрос" in t or "3anpoc" in t:
                anchor_y = w.top
                break

        if anchor_y is None:
            return None

        # Берем правую часть той же полосы и OCR только по желтым символам
        h, w = screen_bgr.shape[:2]
        y1 = max(0, int(anchor_y / 2) - 25)
        y2 = min(h, int(anchor_y / 2) + 100)
        roi = screen_bgr[y1:y2, int(w * 0.45) : w]
        if roi.size == 0:
            return None

        mask = self.yellow_text_mask(roi)
        cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,G"
        try:
            txt = pytesseract.image_to_string(mask, lang=self.ocr_languages, config=cfg)
        except pytesseract.TesseractError:
            txt = pytesseract.image_to_string(mask, lang="eng", config=cfg)
        return self.parse_price(txt)

    def find_p2(self, screen_bgr: np.ndarray) -> Optional[float]:
        prep = self.preprocess(screen_bgr)
        words = self._ocr_data(prep)

        buy_rows = []
        for w in words:
            t = w.text.lower()
            if "купит" in t or "kynut" in t or "kупит" in t:
                buy_rows.append(w)

        if not buy_rows:
            return None

        top_buy = min(buy_rows, key=lambda x: x.top)

        # В исходных координатах OCR выполнялся на изображении x2, поэтому делим на 2
        y_center = max(0, int((top_buy.top + top_buy.height // 2) / 2))
        h, w = screen_bgr.shape[:2]
        y1 = max(0, y_center - 45)
        y2 = min(h, y_center + 45)
        x2 = max(1, int((top_buy.left - 20) / 2))
        roi = screen_bgr[y1:y2, 0:x2]
        if roi.size == 0:
            return None

        mask = self.yellow_text_mask(roi)
        cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,G"
        txt = pytesseract.image_to_string(mask, lang="eng", config=cfg)
        return self.parse_price(txt)

    def grab_screen(self) -> np.ndarray:
        with mss.mss() as sct:
            shot = np.array(sct.grab(sct.monitors[1]))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    def run(self) -> None:
        while True:
            p1 = None
            p2 = None
            try:
                screen = self.grab_screen()
                p1 = self.find_p1(screen)
                p2 = self.find_p2(screen)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] ошибка обработки кадра: {exc}")

            print(f"p1={p1} | p2={p2}")
            time.sleep(self.interval_sec)


if __name__ == "__main__":
    PriceMonitor(interval_sec=1.0).run()
