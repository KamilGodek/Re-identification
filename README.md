# People Counter Project

The People Counter project is an advanced system for counting people based on video footage from a camera. It utilizes the YOLO model for object detection, the SORT algorithm for tracking, and the SIFT algorithm for re-identification of individuals. The system tracks people in video frames, counts them, and saves snapshots of detected individuals.

## Project Structure

- **People_counter/People_counter.py**: Main script for counting people and re-identification.
- **Pic_people/**: Directory where snapshots of detected individuals are stored.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- cvzone
- torch
- ultralytics (YOLO)
- sort
- glob

## Key Features

- **extract_sift_features(image)**: Extracts SIFT features from an image.
- **compare_sift_features(descriptors1, descriptors2)**: Compares SIFT features between two images.
- **reidentify_people(image_path, output_folder)**: Re-identifies people based on SIFT features.

## Additional Information

- The script automatically creates folders for snapshots of detected individuals.
- Uses GPU (CUDA) if available, otherwise, computations are performed on the CPU.
- Displays the count of people crossing lines on the video screen.

## Usage

To run the project, ensure all dependencies are installed, and execute the main script:

```bash
python People_counter/People_counter.py
```

Make sure to update the paths for the YOLO model, video file, and other resources as per your setup. The script will process the video, detect, track, and re-identify people, and display the results in a window while saving snapshots in the specified directory.


![counter](https://github.com/KamilGodek/Projekt_SystemyWizyjne/assets/135075598/034cd398-d485-48fa-931d-4a497e6454a7)
