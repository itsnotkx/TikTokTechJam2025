# YOLO Auto-Blur Video Processor

This project provides a Streamlit-based web application for automatic detection and blurring of sensitive content in videos using a YOLOv8 model. The system is designed for privacy protection, such as anonymizing identity documents, passports, and credit cards.

## Features

- **Automatic Object Detection**: Utilizes a YOLOv8 model finetuned on the [MIDV-500 dataset](https://github.com/fcakyon/midv-500) for robust detection of sensitive content.
- **Multiple Blur Types**: Supports various blur/censoring effects, including:
  - Pixelate
  - Gaussian blur
  - Motion blur
  - Blackout
  - Whiteout
  - Mosaic
- **Object Tracking**: Tracks detected objects across frames for consistent blurring, even when objects move.
- **Temporal Smoothing**: Smooths bounding box positions over time to reduce jitter and improve visual quality.
- **Customizable Parameters**: Easily adjust blur strength, detection confidence, tracking distance, and smoothing factor via the sidebar.
- **Progress Feedback**: Real-time progress bar and status updates during video processing.
- **Downloadable Results**: Download the processed, blurred video directly from the web interface.

## Model Training

- **Base Model**: YOLOv8n-pt
- **Finetuning Dataset**: MIDV-500 (identity documents, license plates, numbers)
- **Training Epochs**: 30

## Usage

1. **Install Requirements**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```sh
   streamlit run backend/app.py
   ```

3. **Upload and Process Video**
   - Upload a video file (supported formats: mp4, avi, mov, mkv, flv, wmv).
   - Adjust detection and blur settings in the sidebar.
   - Click "Process Video" to start anonymization.
   - Download the processed video when complete.

## Advanced Settings

- **Enable Object Tracking**: Maintains object identity across frames for stable blurring.
- **Enable Temporal Smoothing**: Reduces bounding box jitter for smoother results.
- **Tracking Max Distance**: Controls how far an object can move between frames and still be tracked.
- **Smoothing Factor**: Adjusts the responsiveness of temporal smoothing.

## Code Structure

- `YOLOAutoBlur`: Main class for loading the model, detecting objects, and applying blur.
- `BoundingBoxTracker`: Associates detections across frames.
- `TemporalSmoother`: Smooths bounding box coordinates over time.
- Streamlit UI: Provides an interactive web interface for video upload, parameter tuning, and result download.

## Model Weights

- Default model path: `runs/detect/blur_detection_model5/weights/best.pt`
- Update the model path in the sidebar if you use a different checkpoint.

## Citation

- MIDV-500 Dataset: https://github.com/fcakyon/midv-500
- YOLOv8: https://github.com/ultralytics/ultralytics

---

For more details, see backend/app.py.
