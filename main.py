import os
import contextlib
import re
import mss
import cv2
import numpy as np
import pytesseract
import pyautogui
import time
import threading

REFREASH_INTERVAL = 10
PRICE_SHIFT = 0.01

# Глобальный флаг для синхронизации
update_p2_flag = threading.Event()

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
        pyautogui.click(point[0], point[1])
        # print("[ФОН] Нажата кнопка обновления")
        # ПОДНИМАЕМ ФЛАГ:
        update_p2_flag.set()

def check_counters(
        p1_region, 
        p2_region, 
        order_point, 
        cancel_point, 
        input_point, 
        confirm_point):
    # Инициализируем значения текущими данными с экрана
    with mss.mss() as sct:
        p1 = get_value_from_region(sct, p1_region) or 0.0
        p2 = get_value_from_region(sct, p2_region) or 0.0
        
        print(f"Старт мониторинга. p1={p1}, p2={p2}")
        
        while True:
            new_p1 = get_value_from_region(sct, p1_region)
            if new_p1 is not None and new_p1 != p1:
                print(f"[СОБЫТИЕ] p1 изменился: {p1} -> {new_p1}")
                
                if (new_p1 + PRICE_SHIFT < p2):
                    print("!!! Условие выполнено (p1 + shift < p2) !!!")
                    pyautogui.click(order_point)
                    pyautogui.click(input_point)
                    pyautogui.write(str(new_p1 + PRICE_SHIFT), interval=0.1)
                    time.sleep(0.1)  # Короткая пауза перед подтверждением
                    pyautogui.press('enter')
                    pyautogui.click(confirm_point)
                    time.sleep(10)
                    pyautogui.click(cancel_point)
                p1 = new_p1

            if update_p2_flag.is_set():
                # Небольшая задержка, чтобы сайт успел загрузить новые данные после клика
                time.sleep(1) 
                
                new_p2 = get_value_from_region(sct, p2_region)
                if new_p2 is not None and new_p2 != p2:
                    print(f"[СОБЫТИЕ] p2 изменился: {p2} -> {new_p2}")
                    p2 = new_p2
                
                # СБРАСЫВАЕМ ФЛАГ, чтобы не обновлять p2 до следующего клика
                update_p2_flag.clear()

            time.sleep(0.5) # Пауза между проверками

if __name__ == "__main__":
    # 1. Выводим единую инструкцию
    print("\n" + "="*60)
    print("ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫДЕЛЕНИЯ (запомните или смотрите на скриншот):")
    print("-" * 60)
    print("ЭТАП 1:")
    print("  1. Область значнеия запроса на покупку (без G)")
    print("  2. Область значния предложения на покупку (без G)")
    print("  3. Точка кнопки 'ЗАКАЗАТЬ' (в списке)")
    print("  4. Точка кнопки 'ОТМЕНА'")
    print("  5. Точка кнопки 'Только мои запросы' (обновление страницы)")
    print("-" * 60)
    print("ЭТАП 2 (Окно покупки — после нажатия Enter):")
    print("  6. Точка ПОЛЯ ВВОДА цены")
    print("  7. Точка кнопки 'ЗАКАЗАТЬ'")
    print("="*60)
    
    print("\nПодготовьте окна, затем откройте скриншот. Скриншот будет сделан через 3 секунд...\n")
    time.sleep(3) # Даем время развернуть терминал, чтобы он попал в кадр

    # --- ЭТАП 1 ---
    with mss.mss() as sct:
        monitor = sct.monitors[1] 
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)[:, :, :3]
    
    # Последовательный вызов без лишних принтов, чтобы не забивать консоль
    p1_rgn       = get_region(img)
    p2_rgn       = get_region(img)
    order_pnt    = get_click_point(img)
    cancel_pnt   = get_click_point(img)
    refreash_pnt = get_click_point(img)

    # --- ПЕРЕХОД ---
    print("\n>>> Откройте окно покупки на экране и нажмите ENTER в терминале...")
    input()

    # --- ЭТАП 2 ---
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)[:, :, :3]

    input_pnt    = get_click_point(img)
    confirm_pnt  = get_click_point(img)

    print("\n[СОБЫТИЕ] Настройка завершена. Скрипт запущен. Закройте окно запроса. Не трогайте мышь и окно эмулятора\n")

    # Проверяем, что все 7 элементов настроены
    if all([p1_rgn, p2_rgn, order_pnt, cancel_pnt, refreash_pnt, input_pnt, confirm_pnt]):
        
        # Фоновый поток для обновления (рефреша)
        bg_thread = threading.Thread(target=refreash_clicker, args=(refreash_pnt,), daemon=True)
        bg_thread.start()

        try:
            # ПЕРЕДАЕМ ВСЕ ПЕРЕМЕННЫЕ В ФУНКЦИЮ
            check_counters(
                p1_rgn, 
                p2_rgn, 
                order_pnt, 
                cancel_pnt, 
                input_pnt, 
                confirm_pnt
            )
        except KeyboardInterrupt:
            print("\nПрограмма остановлена пользователем.")
    else:
        print("\n[ОШИБКА] Настройка не завершена. Не все области/точки выбраны.")