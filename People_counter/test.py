import os
import numpy as np
import cv2
import cvzone
import math
import torch
from ultralytics import YOLO
from sort import Sort
import glob

# Funkcja do ekstrakcji cech SIFT z obrazu
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Funkcja do porównywania cech SIFT między dwoma obrazami
def compare_sift_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.90 * n.distance:
            good_matches.append([m])
    return len(good_matches)

# Funkcja do re-identyfikacji osób na podstawie cech SIFT
def reidentify_people(image_path, output_folder):
    if os.path.exists(image_path):
        last_image = cv2.imread(image_path)
        return last_image
    else:
        print(f"Plik {image_path} nie istnieje.")
        return None

# W miejscu wywołania funkcji reidentify_people:
output_folder = "E:/Programowanie/Magisterka/Pic_people"  # Przykładowa ścieżka, ustaw ją odpowiednio

# Definicja zmiennych
person_id = 0  # Domyślna wartość dla identyfikatora osoby
frame_count = 0  # Inicjalizacja licznika klatek
last_image_path = None  # Inicjalizacja ścieżki do ostatniego obrazu

last_image = reidentify_people(os.path.join(output_folder, f"person_{person_id}_{frame_count}.jpg"), output_folder)
if last_image is not None:
    _, last_descriptors = extract_sift_features(last_image)

    # Znalezienie innych obrazów w folderze
    other_images_paths = glob.glob(os.path.join(output_folder, '*.jpg'))

    # Re-identyfikacja osób
    for image_path in other_images_paths:
        # Pomijanie porównywania z samym sobą
        if image_path == last_image_path:
            continue

        # Wczytanie obrazu
        other_image = cv2.imread(image_path)
        _, other_descriptors = extract_sift_features(other_image)

        # Porównanie cech SIFT
        num_matches = compare_sift_features(last_descriptors, other_descriptors)

        # Wyświetlanie wyników
        print(f"Number of matches between {last_image_path} and {image_path}: {num_matches}")

# Sprawdzenie dostępności CUDA (GPU) i ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Inicjalizacja modelu YOLO i przeniesienie go na urządzenie GPU
model = YOLO('E:/Programowanie/Magisterka/file_storage/YOLO-Size/yolov8l.pt').to(device)

# Inicjalizacja kamery wideo, z której będą pobierane klatki
cap = cv2.VideoCapture('E:/Programowanie/Magisterka/file_storage/Imagies/Nienazwany.mp4')

# Lista dla nazw klas obiektów
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Wczytanie maski - obrazu pomocniczego do zdefiniowania obszaru detekcji
mask = cv2.imread('E:/Programowanie/Magisterka/file_storage/Imagies/maseczka.png')

# Inicjalizacja SORT - algorytmu do śledzenia obiektów
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Definicja linii, którą obiekty przekraczają, aby były zliczane
linesDown = [210, 320, 295, 320]
linesUp = [675, 320, 760, 320]

# Lista przechowująca unikalne identyfikatory zliczonych obiektów
totalCountUp = []
totalCountDown = []

# Definicja katalogu, w którym będą przechowywane foldery z zrzutami osób
output_folder = "E:/Programowanie/Magisterka/Pic_people"  # Zmień ścieżkę według własnych potrzeb

# Licznik klatek
frame_count = 0

# Główna pętla programu
while True:
    # Odczytanie klatki z kamery
    success, img = cap.read()
    if not success:
        break

    # Stworzenie kopii obrazu bez linii i ID
    img_clean = img.copy()

    # Konwersja klatki do tensora PyTorch i przeniesienie jej na urządzenie GPU
    img_tensor = torch.from_numpy(img).to(device)

    # Stworzenie regionu zainteresowania (ROI) na podstawie maski
    imgRegion = cv2.bitwise_and(img, mask)

    # Dodanie grafiki na tło, na background licznika
    imgGraphic = cv2.imread('E:/Programowanie/Magisterka/file_storage/Imagies/strzalki.png', cv2.IMREAD_UNCHANGED)
    imgGraphic_tensor = torch.from_numpy(imgGraphic).to(device)
    img = cvzone.overlayPNG(img_tensor.cpu().numpy(), imgGraphic_tensor.cpu().numpy(), (450,120))

    # Wywołanie modelu YOLO na zdefiniowanym regionie
    resultes = model(imgRegion, stream=True)

    # Inicjalizacja pustej tablicy dla wyników detekcji
    detections = np.empty((0, 5))

    # Przetwarzanie wyników detekcji z modelu YOLO
    for r in resultes:
        boxes = r.boxes

        # Przetwarzanie poszczególnych detekcji
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Sprawdzenie, czy obiekt jest osobą i pewność detekcji przekracza 0.3
            if (currentClass == "person" and conf > 0.3):
                # Dodanie informacji o detekcji do tablicy
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Przeniesienie wyników detekcji na urządzenie CPU
    detections_tensor = torch.from_numpy(detections).to(device)

    # Aktualizacja tracker'a na podstawie wyników detekcji
    resultesTracker = tracker.update(detections_tensor.cpu().numpy())

    # Rysowanie linii do zliczania
    cv2.line(img, (linesDown[0], linesDown[1]), (linesDown[2], linesDown[3]), (0, 0, 255), 2)
    cv2.line(img, (linesUp[0], linesUp[1]), (linesUp[2], linesUp[3]), (0, 0, 255), 2)

    # Przetwarzanie wyników tracker'a
    for result in resultesTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Rysowanie prostokąta wokół zidentyfikowanego obiektu
        cvzone.cornerRect(img, (x1, y1, w, h), l=3, rt=1, colorR=(255, 0, 255))

        # Wypisanie identyfikatora obiektu nad głową
        cvzone.putTextRect(img, f'ID: {int(id)}', (x1, y1 - 10), scale=1, thickness=1, offset=1, colorR=(255, 0, 255), colorT=(255, 255, 255))

        # Wyznaczenie współrzędnych środka obiektu
        cx, cy = x1 + w // 2, y1 + h // 2

        # Rysowanie kółka w miejscu środka obiektu
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Sprawdzenie, czy obiekt przekracza linię zliczania w dół
        if linesDown[0] < cx < linesDown[2] and linesDown[1] - 15 < cy < linesDown[1] + 15:
            if id not in totalCountDown:
                totalCountDown.append(id)
                # Rysowanie linii w przypadku zliczenia obiektu
                cv2.line(img, (linesDown[0], linesDown[1]), (linesDown[2], linesDown[3]), (0, 255, 0), 5)

        # Sprawdzenie, czy obiekt przekracza linię zliczania w górę
        if linesUp[0] < cx < linesUp[2] and linesUp[1] - 15 < cy < linesUp[1] + 15:
            if id not in totalCountUp:
                totalCountUp.append(id)
                # Rysowanie linii w przypadku zliczenia obiektu
                cv2.line(img, (linesUp[0], linesUp[1]), (linesUp[2], linesUp[3]), (0, 255, 0), 5)

        # Tworzenie folderu dla każdego embeddingu wykrytej osoby
        person_folder = os.path.join(output_folder, f"person_{int(id)}")
        os.makedirs(person_folder, exist_ok=True)

        # Zapisywanie zrzutów osób wykrytych w odpowiednich folderach
        frame_count += 1
        cv2.imwrite(os.path.join(person_folder, f"person_{int(id)}_{frame_count}.jpg"), img_clean[y1:y2, x1:x2])

        # Re-identyfikacja osób na podstawie cech SIFT
        reidentify_people(os.path.join(output_folder, f"person_{int(id)}_{frame_count}.jpg"), output_folder)

    # Wypisanie liczby zliczonych obiektów na obrazie
    cvzone.putTextRect(img, f'Count Down: {len(totalCountDown)}', (50, 50), scale=1.5, thickness=1, colorT=(255, 255, 255),
                        colorR=(186, 82, 79), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0))

    cvzone.putTextRect(img, f'Count Up: {len(totalCountUp)}', (700, 50), scale=1.5 , thickness=1, colorT=(255, 255, 255),
                        colorR=(186, 82, 79), font=cv2.FONT_HERSHEY_PLAIN, offset=5, border=None, colorB=(0, 255, 0))

    # Wyświetlenie obrazu
    cv2.imshow("Image", img)

    # Oczekiwanie na naciśnięcie klawisza, sprawdzenie czy to klawisz ESC (27 to kod klawisza ESC)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Zamknięcie okna po wyjściu z pętli
cv2.destroyAllWindows()
cap.release()