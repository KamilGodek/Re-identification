## Re-identification
# Re-identification
Opis projektu
Projekt People Counter to zaawansowany system do zliczania osób na podstawie materiału wideo z kamery. Wykorzystuje model YOLO do detekcji obiektów, algorytm SORT do śledzenia oraz algorytm SIFT do re-identyfikacji osób. System śledzi osoby w klatkach wideo, zlicza je, a także zapisuje zrzuty wykrytych osób.

Struktura projektu
People_counter/People_counter.py: Główny skrypt realizujący zliczanie osób oraz re-identyfikację.
Pic_people/: Katalog, w którym przechowywane są zrzuty wykrytych osób.
Wymagania
Python 3.8+
OpenCV
NumPy
cvzone
torch
ultralytics (YOLO)
sort
glob



Kluczowe funkcje
extract_sift_features(image): Ekstrakcja cech SIFT z obrazu.
compare_sift_features(descriptors1, descriptors2): Porównywanie cech SIFT między dwoma obrazami.
reidentify_people(image_path, output_folder): Re-identyfikacja osób na podstawie cech SIFT.
Informacje dodatkowe
Skrypt automatycznie tworzy foldery dla zrzutów wykrytych osób.
Używa GPU (CUDA) jeśli jest dostępne, w przeciwnym razie obliczenia są wykonywane na CPU.
Wyświetla licznik osób przekraczających linie na ekranie wideo.
