import re
import mss
import cv2
import numpy as np
import pytesseract
import pyautogui
import time
import threading

REFREASH_INTERVAL = 10 
PRICE_SHIFT = 0.1 

# Если Tesseract не в PATH, раскомментируйте строку ниже и укажите путь:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_region(img):
    cv2.namedWindow("Выделите область", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Выделите область", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    r = cv2.selectROI("Выделите область", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = r
    return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)} if w > 0 else None

def get_click_point(img):
    point = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point.append((x, y))

    cv2.namedWindow("Выберите точку", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Выберите точку", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Выберите точку", mouse_callback)
    while not point:
        cv2.imshow("Выберите точку", img)
        if cv2.waitKey(1) & 0xFF == 27: break
    cv2.destroyAllWindows()
    return point[0] if point else None

def get_value_from_region(sct, coords):
    # Захват области
    sct_img = sct.grab(coords)
    roi_img = np.array(sct_img)
    
    # Предобработка для OCR
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGRA2GRAY)
    # Порог (Threshold) делает текст черным на белом фоне, что сильно улучшает точность
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789."
    txt = pytesseract.image_to_string(thresh, config=cfg)
    
    # Ищем число с точкой или просто число
    m = re.search(r"\d+(\.\d+)?", txt)
    return float(m.group()) if m else None

def refreash_clicker(point):
    print(f"[ФОН] Поток кликера запущен на точку {point}")
    while True:
        time.sleep(REFREASH_INTERVAL)
        pyautogui.click(point[0], point[1])
        print("[ФОН] Нажата кнопка обновления")

def check_counters(p1_region, p2_region):
    # Инициализируем значения текущими данными с экрана
    with mss.mss() as sct:
        p1 = get_value_from_region(sct, p1_region) or 0.0
        p2 = get_value_from_region(sct, p2_region) or 0.0
        
        print(f"Старт мониторинга. p1={p1}, p2={p2}")
        
        while True:
            new_p1 = get_value_from_region(sct, p1_region)
            new_p2 = get_value_from_region(sct, p2_region)

            if new_p1 is not None and new_p1 != p1:
                print(f"[СОБЫТИЕ] p1 изменился: {p1} -> {new_p1}")
                # Ваша логика здесь
                if (new_p1 + PRICE_SHIFT < p2):
                    print("!!! Условие выполнено (p1 + shift < p2) !!!")
                p1 = new_p1

            if new_p2 is not None and new_p2 != p2:
                print(f"[СОБЫТИЕ] p2 изменился: {p2} -> {new_p2}")
                p2 = new_p2

            time.sleep(0.5) # Пауза между проверками

if __name__ == "__main__":
    with mss.mss() as sct:
        monitor = sct.monitors[1] 
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)[:, :, :3]
    
    p1_regn = get_region(img)
    p2_regn = get_region(img)
    refr_point = get_click_point(img)

    if p1_regn and p2_regn and refr_point:
        # ИСПРАВЛЕНО: передаем функцию и аргументы раздельно
        bg_thread = threading.Thread(target=refreash_clicker, args=(refr_point,), daemon=True)
        bg_thread.start()

        try:
            check_counters(p1_regn, p2_regn)
        except KeyboardInterrupt:
            print("Программа остановлена.")
    else:
        print("Настройка отменена или не завершена.")