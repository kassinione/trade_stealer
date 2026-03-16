# --- БИБЛИОТЕКИ ---
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pytesseract
import threading
import pyautogui
import keyboard
import logging
import time
import sys
import mss
import cv2
import re

# --- КОНФИГУРАЦИЯ ---
STOP_BIND: str = "f8"
LOG_FILE:  str = "debug.log"

REFREASH_INTERVAL:  int = 5
OCR_MAX_EMPTY_TIME: int = 10

PRICE_INCREMENT: float = 0.10  # На сколько перебиваем конкурента (p1)
MIN_PRICE_GAP:   float = 0.10  # Минимальная разница между запросом и предложением

# --- СКРИПТ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)

logger = logging.getLogger("TradeBot")

update_p2_flag = threading.Event()

refreash_allowed = threading.Event()
refreash_allowed.set() 

stop_flag = threading.Event()

mouse_lock = threading.Lock()

Region = Dict[str, int]
Point = Tuple[int, int]

def stop():
    logger.info("SYSTEM: Stop hotkey pressed. Shutting down...")
    stop_flag.set()

# Выбор области на экране через OpenCV ROI.
def get_region(img: np.ndarray) -> Optional[Region]:
    window_name = "Selection: Draw Rectangle and Press SPACE/ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    r = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    x, y, w, h = r
    if w > 0 and h > 0:
        # logger.debug(f"REGION_CONFIG: Defined at x={x}, y={y}, w={w}, h={h}")
        return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
    
    return None

# Выбор точки клика на экране.
def get_click_point(img: np.ndarray) -> Optional[Point]:
    point: List[Point] = []
    window_name = "Selection: Click Point and Press ESC"

    def mouse_callback(event: int, x: int, y: int, flags: Any, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point.append((x, y))
            # logger.debug(f"POINT_CONFIG: Selected point ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while not point:
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == 27: break
    
    cv2.destroyAllWindows()

    return point[0] if point else None

# Захват области и распознавание числа (OCR) с двойной проверкой.
def get_value_from_region(sct: mss.mss, coords: Region, retry: bool = True) -> Optional[float]:
    def read_once() -> Optional[float]:
        try:
            sct_img = sct.grab(coords)
            # Конвертируем в формат OpenCV
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 1. Увеличиваем изображение в 3 раза (интерполяция CUBIC лучше всего для текста)
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # 2. Переводим в ЧБ
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Убираем шумы (размытие)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # 4. Метод Оцу: автоматически подбирает порог
            # Если текст светлее фона, используем THRESH_BINARY_INV (делаем текст черным на белом)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 5. Добавляем пустую белую рамку (Padding) — это ОЧЕНЬ помогает Tesseract
            border_size = 10
            thresh = cv2.copyMakeBorder(
                thresh, 
                top=border_size, bottom=border_size, 
                left=border_size, right=border_size, 
                borderType=cv2.BORDER_CONSTANT, 
                value=[255, 255, 255]
            )

            # Настройка OCR
            # psm 7 — одна строка, psm 6 — единый блок текста
            cfg = "--psm 7 -c tessedit_char_whitelist=0123456789."
            txt = pytesseract.image_to_string(thresh, config=cfg)

            # Чистим результат от лишних пробелов/символов
            txt = txt.strip().replace(" ", "")
            
            m = re.search(r"\d+(\.\d+)?", txt)
            if m:
                return float(m.group())

        except Exception as e:
            logger.error(f"OCR_ERROR: {e}")
        return None

    v1 = read_once()
    v2 = read_once()

    if v1 is not None and v2 is not None and abs(v1 - v2) < 0.01:
        # logger.debug(f"OCR_READ: Confirmed value {v2}")
        return v2

    if retry:
        logger.debug("OCR_RETRY: values mismatch, retrying...")
        return get_value_from_region(sct, coords, retry=False)

    logger.debug("OCR_FAIL: unable to confirm value")
    return None

# Фоновый поток для циклического обновления страницы.
def refreash_clicker(point: Point) -> None:
    # logger.info(f"THREAD_START: Refreash clicker active on point {point}")
    
    while not stop_flag.is_set():
        # logger.debug("SIGNAL: Refreash triggered")
        
        refreash_allowed.wait() # ждём пока refreash снова разрешат
        time.sleep(0.1)

        with mouse_lock:
            if not refreash_allowed.is_set(): continue

            pyautogui.click(point[0], point[1], clicks=2, interval=0.2)
            update_p2_flag.set()

        time.sleep(REFREASH_INTERVAL)

def check_counters(
        p1_region:      Region, 
        p2_region:      Region, 
        order_point:    Point, 
        cancel_point:   Point, 
        input_point:    Point, 
        refreash_point: Point,
        confirm_point:  Point,
        x_point:        Point
        ) -> None:
    
    # Основной цикл мониторинга и принятия решений.
    with mss.mss() as sct:
        p1: float = get_value_from_region(sct, p1_region) or 0.0
        p2: float = get_value_from_region(sct, p2_region) or 0.0
        my_last_price: float = 0.0 
        last_p2_success_time = time.time()
        # tried_fix_refreash = False

        logger.info(f"SYSTEM: Monitoring starts.")
        
        while not stop_flag.is_set():
            current_time = time.time()
            time_since_last_p2 = current_time - last_p2_success_time

            # Если P2 долго не читается — попытка исправить, если не помогло — аварийный выход
            if time_since_last_p2 > OCR_MAX_EMPTY_TIME:
            
                # if not tried_fix_refreash:
                #     with mouse_lock:
                #         pyautogui.click(refreash_point)
                #         time.sleep(1.0)
                #         logger.critical("RECOVERY: Попытка исправить кнопку обновления")
                #     tried_fix_refreash = True
                #     last_p2_success_time = time.time() - (OCR_MAX_EMPTY_TIME / 2)
                #     continue

                # else:
                logger.critical("FATAL: Область P2 пуста или не читается слишком долго!")
                stop_flag.set()
                break

            new_p1 = get_value_from_region(sct, p1_region)
            
            if new_p1 is not None and new_p1 != p1:
                # Защита от самоперебивания
                if abs(new_p1 - my_last_price) < 0.01:
                    # logger.debug(f"SKIP: Detected own price {new_p1}. Ignoring.")
                    p1 = new_p1
                    continue
                
                # logger.debug(f"CHECK: Target: {target_price} | Max Allowed: {max_allowed_price}")
                
                target_price = new_p1 + PRICE_INCREMENT

                # Условие для создания запроса
                if (p2 - target_price > MIN_PRICE_GAP):
                    refreash_allowed.clear()   # остановить обновление
        
                    logger.warning(f"ACTION: Создаем заказ. Наша цена: {target_price} < Предложение: {p2}")
                    
                    with mouse_lock:
                        try:
                            pyautogui.click(order_point)
                            time.sleep(0.1)
                            pyautogui.click(input_point)
                            
                            pyautogui.write(str(target_price), interval=0.05)
                            time.sleep(0.1)
                            pyautogui.press('enter')
                            
                            pyautogui.click(confirm_point)
                            
                            my_last_price = target_price
                            logger.info(f"SUCCESS: Order placed at {target_price}")
                            
                            time.sleep(0.5)
                            pyautogui.click(cancel_point)

                        except Exception as e:
                            logger.error(f"RUNTIME_ERROR: Action sequence failed: {e}")

                        finally:
                            time.sleep(0.1)
                            refreash_allowed.set()   # снова включить refreash
                
                p1 = new_p1

            # Обновление p2 (цены предложения) по флагу из потока рефреша
            if update_p2_flag.is_set():
                time.sleep(1.2)    # Задержка для загрузки станицы
                new_p2 = get_value_from_region(sct, p2_region)
                
                if new_p2 is not None:
                    p2 = new_p2
                    last_p2_success_time = time.time()
                    # tried_fix_refreash = False
                    # logger.info(f"SYNC: Base price p2 updated to {p2}")
                
                update_p2_flag.clear()

            time.sleep(0.2)
            
        logger.info("SYSTEM: Monitoring stopped.")
        sys.exit(0)

if __name__ == "__main__":

    keyboard.add_hotkey(STOP_BIND, stop)

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*17 + "ИНСТРУКЦИЯ ПО НАСТРОЙКЕ СКРИПТА" + " "*10 + "║")
    print("╠" + "═"*58 + "╣")
    print("║ " + "ЭТАП 1: ГЛАВНЫЙ ЭКРАН (выделяем на первом скриншоте)" + " " * 5 + "║")
    print("║ " + "-"*56 + " ║")
    print("║ [🔲] 1. Область ЦЕНЫ ЗАПРОСА (рамка без значка 'G')      ║")
    print("║ [🔲] 2. Область ЦЕНЫ ПРЕДЛОЖЕНИЯ (рамка без значка 'G')  ║")
    print("║ [🎯] 3. Кнопка 'ЗАКАЗАТЬ' (центр кнопки)                 ║")
    print("║ [🎯] 4. Кнопка 'ОТМЕНА' (центр под кнопкой ЗАКАЗАТЬ)     ║")
    print("║ [🎯] 5. Кнопка 'ТОЛЬКО МОИ ЗАПРОСЫ' (центр кнопки)       ║")
    print("║" + " "*58 + "║")
    print("║ " + "ЭТАП 2: ОКНО ЗАКАЗА (после нажатия Enter в консоли)" + " " * 6 + "║")
    print("║ " + "-"*56 + " ║")
    print("║ [🎯] 6. ПОЛЕ ВВОДА цены (центр поля)                     ║")
    print("║ [🎯] 7. Кнопка 'ЗАКАЗАТЬ' (центр кнопки)                 ║")
    print("║ [🎯] 8. Кнопка закрытия окна 'X' (правый верхний угол)   ║")
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
        x_pnt        = get_click_point(img)

        print("[!] Настройка завершена. Скрипт будет запущен через 3 секунды...")
        print("[!] Не перемещайте окно эмулятора.")
        print("[!] Для остановки скрипта нажмите F8.")

        if all([p1_rgn, p2_rgn, order_pnt, cancel_pnt, refreash_pnt, input_pnt, confirm_pnt, x_pnt]):
                # logger.info("SYSTEM: Configuration complete. Starting threads...")
                
                pyautogui.click(x_pnt)

                bg_thread = threading.Thread(target=refreash_clicker, args=(refreash_pnt,), daemon=True)
                bg_thread.start()
                
                check_counters(p1_rgn, p2_rgn, order_pnt, cancel_pnt, input_pnt, refreash_pnt, confirm_pnt, x_pnt)
        else:
            logger.critical("SYSTEM: Configuration failed. Not all points/regions were selected.")
            
    except KeyboardInterrupt:
        logger.info("SYSTEM: Shutdown by user.")

    except Exception as e:
        logger.critical(f"SYSTEM: Critical failure: {e}")