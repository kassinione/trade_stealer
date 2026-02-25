import re
import mss
import cv2
import numpy as np
import pytesseract
import pyautogui
import time
import threading

REFREASH_INTERVAL = 10  # Интервал нажатия кнопки обновления в секундах
PRICE_SHIFT       = 0.1 # Минимальная разница между запросом и предложением ( < p2 - p1 )

def get_region(img):
    # Полноэкранное окно для точного выбора
    cv2.namedWindow("Выделите область", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Выделите область", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    r = cv2.selectROI("Выделите область", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = r
    if w == 0 or h == 0:
        return None

    roi_coords = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}

    return roi_coords

def get_click_point(img):
    point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point.append((x, y))

    cv2.namedWindow("Выберите точку", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Выберите точку", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.setMouseCallback("Выберите точку", mouse_callback)

    while True:
        cv2.imshow("Выберите точку", img)
        if point:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

    return point[0] if point else None

def get_value_from_region(coords):
    roi_coords = coords
    
    # Сразу получаем изображение выбранной области
    with mss.mss() as sct:
        roi_img = np.array(sct.grab(roi_coords))
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGRA2BGR)
    
    if roi_img is None:
        print("Область не выбрана")
        exit()

    # Перевод в серый
    g = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Лёгкое размытие для улучшения OCR
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # Увеличение
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # OCR: только цифры и точка
    cfg = "--psm 8 -c tessedit_char_whitelist=0123456789."
    txt = pytesseract.image_to_string(g, config=cfg)

    m = re.search(r"\d+\.\d+", txt)
    val = float(m.group()) if m else None
    
    return val

def click_at(x, y):
    pyautogui.click(x, y)

def refreash_clicker(point):
    """Функция, которая просто кликает раз в 10 секунд в фоне"""
    while True:
        print("[ФОН] Нажатие кнопки...")
        click_at(point)
        time.sleep(INTERVAL)

def check_counters(p1_region, p2_region):
    """Основная логика слежки за счетчиками"""
    p1 = 0
    p2 = 0
    
    print("Программа запущена. Слежу за счетчиками...")
    
    while True:
        # 1. Получаем новые значения (эмуляция)
        new_p1 = get_value_from_region(p1_region)
        new_p2 = get_value_from_region(p2_region)

        # 2. Проверяем изменения
        if new_p1 != p1:
            print("[СОБЫТИЕ] Счетчик А изменился! Запуск проверки...")
            
            # Выполняем проверку и нажатие
            # do_logic_click()
            p1 = new_p1
            
        # Небольшая пауза, чтобы не нагружать процессор на 100%
        time.sleep(0.1)

if __name__ == "__main__":
    # Запускаем фоновый кликер как "демон" 
    # (он закроется автоматически при выходе из основной программы)
    bg_thread = threading.Thread(target=refreash_clicker, daemon=True)
    bg_thread.start()

    # Запускаем основную слежку в главном потоке
    try:
        check_counters()
    except KeyboardInterrupt:
        print("Программа остановлена.")
