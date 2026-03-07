import re
import mss
import cv2
import numpy as np
import pytesseract
import pyautogui
import time
import threading
import logging
from typing import Dict, Tuple, Optional, List, Any

# --- КОНФИГУРАЦИЯ ---
REFREASH_INTERVAL: int = 10
PRICE_SHIFT: float = 0.1
LOG_FILE: str = "debug.log"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)

logger = logging.getLogger("TradeBot")
# Глобальный флаг для синхронизации
update_p2_flag = threading.Event()

Region = Dict[str, int]
Point = Tuple[int, int]

def get_region(img: np.ndarray) -> Optional[Region]:
    """Выбор области на экране через OpenCV ROI."""
    window_name = "Selection: Draw Rectangle and Press SPACE/ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    r = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    x, y, w, h = r
    if w > 0 and h > 0:
        logger.debug(f"REGION_CONFIG: Defined at x={x}, y={y}, w={w}, h={h}")
        return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
    return None

def get_click_point(img: np.ndarray) -> Optional[Point]:
    """Выбор точки клика на экране."""
    point: List[Point] = []
    window_name = "Selection: Click Point and Press ESC"

    def mouse_callback(event: int, x: int, y: int, flags: Any, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point.append((x, y))
            logger.debug(f"POINT_CONFIG: Selected point ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while not point:
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == 27: break
    
    cv2.destroyAllWindows()
    return point[0] if point else None

def get_value_from_region(sct: mss.mss, coords: Region) -> Optional[float]:
    """Захват области и распознавание числа (OCR)."""
    try:
        sct_img = sct.grab(coords)
        roi_img = np.array(sct_img)
        
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGRA2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        cfg = "--psm 7 -c tessedit_char_whitelist=0123456789."
        txt = pytesseract.image_to_string(thresh, config=cfg)
        
        m = re.search(r"\d+(\.\d+)?", txt)
        if m:
            val = float(m.group())
            logger.debug(f"OCR_READ: Recognized value {val}")
            return val
    except Exception as e:
        logger.error(f"OCR_ERROR: Failed to read region {coords}. Details: {e}")
    return None

def refreash_clicker(point: Point) -> None:
    """Фоновый поток для циклического обновления страницы."""
    logger.info(f"THREAD_START: Refresh clicker active on point {point}")
    while True:
        time.sleep(REFREASH_INTERVAL)
        logger.debug("SIGNAL: Refresh triggered")
        pyautogui.click(point[0], point[1])
        pyautogui.click(point[0], point[1])
        update_p2_flag.set()

def check_counters(
        p1_region: Region, 
        p2_region: Region, 
        order_point: Point, 
        cancel_point: Point, 
        input_point: Point, 
        confirm_point: Point) -> None:
    """Основной цикл мониторинга и принятия решений."""
    
    with mss.mss() as sct:
        p1: float = get_value_from_region(sct, p1_region) or 0.0
        p2: float = get_value_from_region(sct, p2_region) or 0.0
        my_last_price: float = 0.0 
        
        logger.info(f"MONITOR_START: Initial states [p1: {p1}, p2: {p2}]")
        
        while True:
            new_p1 = get_value_from_region(sct, p1_region)
            
            if new_p1 is not None and new_p1 != p1:
                # Защита от самоперебивания
                if abs(new_p1 - my_last_price) < 0.01:
                    logger.debug(f"SKIP: Detected own price {new_p1}. Ignoring.")
                    p1 = new_p1
                    continue

                logger.info(f"EVENT: p1 price change detected [{p1} -> {new_p1}]")
                
                # Условие для входа в сделку
                if (new_p1 + PRICE_SHIFT < p2):
                    target_price = round(new_p1 + PRICE_SHIFT, 2)
                    logger.warning(f"ACTION: Outbidding! Target: {target_price} | Gap: {round(p2 - target_price, 2)}")
                    
                    try:
                        pyautogui.click(order_point)
                        time.sleep(0.2)
                        pyautogui.click(input_point)
                        
                        pyautogui.write(str(target_price), interval=0.05)
                        time.sleep(0.1)
                        pyautogui.press('enter')
                        
                        pyautogui.click(confirm_point)
                        
                        my_last_price = target_price
                        logger.info(f"SUCCESS: Order placed at {target_price}")
                        
                        time.sleep(1.0)
                        pyautogui.click(cancel_point)
                    except Exception as e:
                        logger.error(f"RUNTIME_ERROR: Action sequence failed: {e}")
                
                p1 = new_p1

            # Обновление p2 (цены предложения) по флагу из потока рефреша
            if update_p2_flag.is_set():
                time.sleep(1.5) # Задержка для загрузки станицы
                new_p2 = get_value_from_region(sct, p2_region)
                
                if new_p2 is not None and new_p2 != p2:
                    p2 = new_p2
                    logger.info(f"SYNC: Base price p2 updated to {p2}")
                
                update_p2_flag.clear()

            time.sleep(0.2)

if __name__ == "__main__":
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*17 + "ИНСТРУКЦИЯ ПО НАСТРОЙКЕ СКРИПТА" + " "*10 + "║")
    print("╠" + "═"*58 + "╣")
    print("║ " + "ЭТАП 1: ГЛАВНЫЙ ЭКРАН (выделяем на первом скриншоте)" + " " * 5 + "║")
    print("║ " + "-"*56 + " ║")
    print("║ [🔲] 1. Область ЦЕНЫ ЗАПРОСА (рамка без значка 'G')      ║")
    print("║ [🔲] 2. Область ЦЕНЫ ПРЕДЛОЖЕНИЯ (рамка без значка 'G')  ║")
    print("║ [🎯] 3. Кнопка 'ЗАКАЗАТЬ' (в общем списке)               ║")
    print("║ [🎯] 4. Кнопка 'ОТМЕНА' (для сброса окон)                ║")
    print("║ [🎯] 5. Кнопка 'ТОЛЬКО МОИ ЗАПРОСЫ' (для обновления)     ║")
    print("║" + " "*58 + "║")
    print("║ " + "ЭТАП 2: ОКНО ЗАКАЗА (после нажатия Enter в консоли)" + " " * 6 + "║")
    print("║ " + "-"*56 + " ║")
    print("║ [🎯] 6. ПОЛЕ ВВОДА цены (клик в центр поля)              ║")
    print("║ [🎯] 7. Кнопка 'ЗАКАЗАТЬ' (финальное подтверждение)      ║")
    print("╚" + "═"*58 + "╝")
    
    print("\n[!] ПОДГОТОВКА: Разверните игру и терминал.")
    print("[!] Терминал должен быть виден для чтения этой инструкции.")
    print("[!] Делаю скриншот через 3 секунды... Ожидайте открытия окна выбора.\n")
    time.sleep(3)

    # --- ЭТАП 1 ---
    with mss.mss() as sct:
        monitor = sct.monitors[1] 
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)[:, :, :3]
    
    try:
        # Последовательный вызов без лишних принтов, чтобы не забивать консоль
        p1_rgn       = get_region(img)
        p2_rgn       = get_region(img)
        order_pnt    = get_click_point(img)
        cancel_pnt   = get_click_point(img)
        refreash_pnt = get_click_point(img)

        # --- ПЕРЕХОД ---
        print("\n[!] Откройте окно заказа и нажмите ENTER в терминале...")
        input()

        # --- ЭТАП 2 ---
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)[:, :, :3]

        input_pnt    = get_click_point(img)
        confirm_pnt  = get_click_point(img)

        print("[!] Настройка завершена. Скрипт будет запущен после нажатия ENTER в терминале.")
        print("[!] Закройте окно заказа и не перемещайте окно эмулятора")
        input()

        # Проверяем, что все 7 элементов настроены
        if all([p1_rgn, p2_rgn, order_pnt, cancel_pnt, refreash_pnt, input_pnt, confirm_pnt]):
                logger.info("SYSTEM: Configuration complete. Starting threads...")
                
                bg_thread = threading.Thread(target=refreash_clicker, args=(refreash_pnt,), daemon=True)
                bg_thread.start()

                check_counters(p1_rgn, p2_rgn, order_pnt, cancel_pnt, input_pnt, confirm_pnt)
        else:
            logger.critical("SYSTEM: Configuration failed. Not all points/regions were selected.")
            
    except KeyboardInterrupt:
        logger.info("SYSTEM: Shutdown by user.")
    except Exception as e:
        logger.critical(f"SYSTEM: Critical failure: {e}")