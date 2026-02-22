# trade_stealer

Скрипт `monitor_prices.py` раз в секунду делает скриншот экрана, ищет два UI-блока через OCR/текстовые якоря и выводит цены:

- `p1` — цена справа от блока `Запросов на покупку ...`
- `p2` — верхняя цена слева от самой верхней кнопки `КУПИТЬ`

## Установка

```bash
pip install mss opencv-python pytesseract numpy
```

Также должен быть установлен движок Tesseract OCR в системе.

## Запуск

```bash
python monitor_prices.py
```
