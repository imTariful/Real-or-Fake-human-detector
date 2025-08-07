# Real-or-Fake-human-detector

This project implements a YOLOv8-based deepfake detection system to classify human faces in real-time webcam video as **"real"** or **"fake."** It includes a complete pipeline for data collection, dataset preparation, model training, and real-time detection. Designed for applications in security, media authenticity verification, and digital forensics, the system leverages computer vision and machine learning to combat deepfake content effectively.
Features

**Real-Time Face Classification:** Processes live webcam video, detecting faces and labeling them as "REAL" (green) or "FAKE" (red) with bounding boxes, using a confidence threshold of 0.7.

**Data Collection:** Captures high-quality (sharp) face images from a webcam, ensuring robust training data with blur detection (Laplacian variance threshold of 20).

**Dataset Preparation:** Organizes data into training (70%), validation (20%), and test (10%) sets, generating a YOLO-compatible data.yaml configuration file.

**Model Training:** Fine-tunes a pretrained YOLOv8x model for accurate face detection and classification over 15 epochs with a batch size of 16.

**Customizable Parameters:** Adjustable settings for confidence thresholds, image sizes (640x480 for training, 416x416 for detection), and offsets for bounding box expansion.

**FPS Display:** Shows real-time frames per second (FPS) on the video feed for performance monitoring.

**Debug Support:** Optional debug mode in data collection for detailed logging of blur scores and detection results.

**Requirements**

**Python:** Version 3.x
**Dependencies:**

**opencv-python:** For webcam access and image processing.

**cvzone:** For drawing bounding boxes and text overlays (required for Main.ipynb).

**ultralytics:** For YOLOv8 model training and inference.

**numpy:** For numerical computations.


**Hardware:**

Webcam for data collection and real-time detection.
Optional GPU for faster model training (specified as device=0 in train_data.ipynb).


**Installation:** pip install opencv-python cvzone ultralytics numpy



**Project Structure**

The project consists of four Jupyter notebooks, each handling a specific stage of the deepfake detection pipeline:

**data_collection.ipynb:**

Captures real face images from a webcam (resolution 640x480).

Uses cvzone.FaceDetectionModule for face detection with a minimum confidence of 0.5.

Applies blur detection (Laplacian variance > 20) to ensure sharp images.

Saves images and YOLO-format annotations (class ID 0 for real faces) to Dataset/DataCollect.

Features offset adjustments for bounding boxes (10% width, 20% height) and debug mode for logging.**


**split_data.ipynb:**

Reads images and labels from Dataset/all.

Splits data into train (70%), validation (20%), and test (10%) sets, creating subdirectories in Dataset/SplitData (train/images, train/labels, etc.).

Generates data.yaml with paths to split datasets and class names (fake, real).

Handles missing file pairs with warnings.


**train_data.ipynb:**

Loads a pretrained YOLOv8x model (yolov8x.pt) for transfer learning.

Trains the model on Dataset/SplitData/dataoffline.yaml for 15 epochs with a batch size of 16 and image size of 640.

Uses GPU (device=0) for training with verbose logging.


**Main.ipynb:**
Performs real-time detection on webcam video (resolution 416x416).

Loads the trained model and processes frames, labeling faces as "real" or "fake" with confidence scores.

Displays FPS and bounding boxes using cvzone (requires installation to avoid ModuleNotFoundError).

Exits on pressing 'q'.



**Usage Instructions**

**Prepare Environment:**
Install dependencies using the command above.
Ensure a webcam is connected and functional.


**Collect Data:**
Run data_collection.ipynb to capture real face images.
Place fake face images and their annotations in Dataset/all (not included in the provided scripts).
Ensure images are sharp and annotations are in YOLO format.


**Split Dataset:**
Run split_data.ipynb to organize data into Dataset/SplitData with train, validation, and test sets.
Verify data.yaml is created correctly in Dataset/SplitData.


**Train Model:**
Run train_data.ipynb to train the YOLOv8 model.
Update MODEL_PATH in Main.ipynb with the path to the trained model (e.g., runs/detect/train/weights/best.pt).


**Run Real-Time Detection:**
Run Main.ipynb to start live detection.
Ensure cvzone is installed to avoid errors.
Press 'q' to exit the video feed.





**Potential Applications**

**Security:** Verify identities in video calls or surveillance systems.
**Media Authenticity:** Detect deepfakes in videos for journalism or social media platforms.
**Digital Forensics:** Analyze visual evidence for authenticity in legal or investigative contexts.

**Limitations**

Accuracy depends on the quality and diversity of the training dataset.

Real-time detection performance may vary based on hardware (e.g., webcam quality, CPU/GPU speed).

The provided pipeline assumes fake face data is available, which is not covered in the data collection script.

Contributing

Contributions are welcome! Please:

**Fork the repository.**
**Create a pull request with detailed descriptions of changes.**
**Ensure code follows PEP 8 guidelines and includes comments.**

**Author**
**Tariful Islam Tarif**
**Data Scientist & AI Engineer**
**Developed for research and educational purposes in deepfake detection.**

