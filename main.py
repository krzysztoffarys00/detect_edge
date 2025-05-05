import cv2
import os

# 🔹 Ścieżki
INPUT_DIR = "assets"  # <- folder z obrazkami wejściowymi
OUTPUT_DIR = "lab_output"  # <- folder gdzie będą zapisywane wyniki

# 🔹 Parametry eksperymentu
RESIZE_WIDTH = 960
RESIZE_HEIGHT = 1280
CANNY_THRESHOLD1 = 75
CANNY_THRESHOLD2 = 200
APPROX_ACCURACY_PERCENT = 0.02
MIN_CONTOUR_AREA = 500

# 🔹 Upewnij się że output folder istnieje
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Znajdź wszystkie pliki jpg
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Znaleziono {len(image_files)} obrazków do przetworzenia.")

for idx, filename in enumerate(image_files):
    print(f"🔹 Przetwarzam ({idx+1}/{len(image_files)}): {filename}")

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Nie udało się wczytać {filename}")
        continue

    # Powiększenie
    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Szarość, Blur, Canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    # Znalezienie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kopia do rysowania
    result = img.copy()

    # Szukanie największego konturu z 4 punktami
    maxArea = 0
    biggest = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, APPROX_ACCURACY_PERCENT * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    # 🔹 Rysowanie:
    # Najpierw wszystkie kontury cienką żółtą linią
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)  # Żółty

    # Potem największy kontur grubą magentową linią
    if len(biggest) != 0:
        cv2.drawContours(result, [biggest], -1, (255, 0, 255), 20)  # Magenta
        print(f"✅ Znaleziono największy kontur 4-punktowy o powierzchni {maxArea}")
    else:
        print("⚠️ Nie znaleziono poprawnego konturu 4-punktowego.")

    # Zapisz wynik do OUTPUT_DIR
    output_path = os.path.join(OUTPUT_DIR, f"out_{idx:03d}.jpg")
    cv2.imwrite(output_path, result)

print("✅ Przetwarzanie zakończone! Sprawdź folder:", OUTPUT_DIR)