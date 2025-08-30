import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from collections import deque
from scipy.spatial.distance import cdist
import math
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BoundingBoxTracker:
    """Simple tracker for bounding boxes across frames using distance-based matching"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def _get_center(self, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, box1: Tuple[float, float, float, float], 
                           box2: Tuple[float, float, float, float]) -> float:
        center1 = self._get_center(box1)
        center2 = self._get_center(box2)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections: List[Tuple[Tuple[float, float, float, float], int]]) -> Dict[int, Tuple[float, float, float, float]]:
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return {track_id: track['box'] for track_id, track in self.tracks.items()}
        
        if len(self.tracks) == 0:
            for box, class_id in detections:
                self.tracks[self.next_id] = {
                    'box': box,
                    'disappeared': 0,
                    'class_id': class_id
                }
                self.next_id += 1
            return {track_id: track['box'] for track_id, track in self.tracks.items()}
        
        track_centers = [self._get_center(track['box']) for track in self.tracks.values()]
        detection_centers = [self._get_center(box) for box, _ in detections]
        
        if len(track_centers) > 0 and len(detection_centers) > 0:
            distance_matrix = cdist(track_centers, detection_centers)
            
            used_detection_indices = set()
            used_track_indices = set()
            track_ids = list(self.tracks.keys())
            
            matches = []
            for i in range(len(track_ids)):
                for j in range(len(detections)):
                    if i not in used_track_indices and j not in used_detection_indices:
                        if distance_matrix[i, j] < self.max_distance:
                            matches.append((i, j, distance_matrix[i, j]))
            
            matches.sort(key=lambda x: x[2])
            
            for track_idx, detection_idx, distance in matches:
                if track_idx not in used_track_indices and detection_idx not in used_detection_indices:
                    track_id = track_ids[track_idx]
                    box, class_id = detections[detection_idx]
                    
                    self.tracks[track_id]['box'] = box
                    self.tracks[track_id]['disappeared'] = 0
                    self.tracks[track_id]['class_id'] = class_id
                    
                    used_track_indices.add(track_idx)
                    used_detection_indices.add(detection_idx)
            
            for i, track_id in enumerate(track_ids):
                if i not in used_track_indices:
                    self.tracks[track_id]['disappeared'] += 1
                    if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                        del self.tracks[track_id]
            
            for j, (box, class_id) in enumerate(detections):
                if j not in used_detection_indices:
                    self.tracks[self.next_id] = {
                        'box': box,
                        'disappeared': 0,
                        'class_id': class_id
                    }
                    self.next_id += 1
        
        return {track_id: track['box'] for track_id, track in self.tracks.items()}


class TemporalSmoother:
    """Temporal smoothing for bounding boxes"""
    
    def __init__(self, window_size: int = 5, smoothing_factor: float = 0.7):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.history = {}
    
    def smooth_boxes(self, tracked_boxes: Dict[int, Tuple[float, float, float, float]]) -> Dict[int, Tuple[float, float, float, float]]:
        smoothed_boxes = {}
        
        for track_id, box in tracked_boxes.items():
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.window_size)
            
            self.history[track_id].append(box)
            
            if len(self.history[track_id]) == 1:
                smoothed_boxes[track_id] = box
            else:
                prev_box = list(self.history[track_id])[-2]
                current_box = box
                
                smoothed_box = tuple(
                    self.smoothing_factor * current + (1 - self.smoothing_factor) * prev
                    for current, prev in zip(current_box, prev_box)
                )
                smoothed_boxes[track_id] = smoothed_box
        
        active_track_ids = set(tracked_boxes.keys())
        inactive_track_ids = set(self.history.keys()) - active_track_ids
        for track_id in inactive_track_ids:
            del self.history[track_id]
        
        return smoothed_boxes


class YOLOAutoBlur:
    def __init__(self, model_path: str = None, data_yaml_path: str = "data.yaml"):
        self.data_yaml_path = data_yaml_path
        self.model_path = model_path
        self.model = None
        self.class_names = self._load_class_names()
        self.tracker = None
        self.smoother = None
        
    def _load_class_names(self) -> dict:
        try:
            if os.path.exists(self.data_yaml_path):
                with open(self.data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                return data['names']
        except Exception as e:
            logger.warning(f"Could not load class names from {self.data_yaml_path}: {e}")
        
        # Default class names
        return {0: 'license_plate', 1: 'number', 2: 'identity_document'}
     
    def load_model(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("No model path provided.")
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def apply_blur(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                   blur_type: str = "pixelate", blur_strength: int = 15) -> np.ndarray:
        blurred_image = image.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            region = image[y1:y2, x1:x2]
            
            if region.size > 0:
                height, width = region.shape[:2]
                
                if blur_type == "gaussian":
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
                
                elif blur_type == "motion":
                    kernel_size = max(blur_strength, 5)
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                    kernel = kernel / kernel_size
                    blurred_region = cv2.filter2D(region, -1, kernel)
                
                elif blur_type == "pixelate":
                    pixel_size = max(blur_strength // 3, 8)
                    small = cv2.resize(region, (width // pixel_size, height // pixel_size), 
                                     interpolation=cv2.INTER_LINEAR)
                    blurred_region = cv2.resize(small, (width, height), 
                                              interpolation=cv2.INTER_NEAREST)
                
                elif blur_type == "black":
                    blurred_region = np.zeros_like(region)
                
                elif blur_type == "white":
                    blurred_region = np.ones_like(region) * 255
                
                elif blur_type == "mosaic":
                    tile_size = max(blur_strength // 2, 6)
                    blurred_region = region.copy()
                    for i in range(0, height, tile_size):
                        for j in range(0, width, tile_size):
                            tile = region[i:i+tile_size, j:j+tile_size]
                            if tile.size > 0:
                                avg_color = np.mean(tile, axis=(0, 1))
                                blurred_region[i:i+tile_size, j:j+tile_size] = avg_color
                
                else:
                    blur_strength = max(blur_strength, 21)
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
                
                blurred_image[y1:y2, x1:x2] = blurred_region
        
        return blurred_image
    
    def detect_and_blur_video(self, video_path: str, output_path: str, 
                              confidence_threshold: float = 0.5, blur_type: str = "pixelate", 
                              blur_strength: int = 15, enable_tracking: bool = True,
                              enable_smoothing: bool = True, tracking_max_distance: float = 100.0,
                              smoothing_factor: float = 0.7, progress_callback=None):
        if not self.model:
            raise ValueError("Model not loaded.")
        
        if enable_tracking:
            self.tracker = BoundingBoxTracker(max_distance=tracking_max_distance)
        
        if enable_smoothing:
            self.smoother = TemporalSmoother(smoothing_factor=smoothing_factor)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                results = self.model(frame, conf=confidence_threshold)
                
                raw_detections = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            raw_detections.append(((x1, y1, x2, y2), class_id))
                
                if enable_tracking and self.tracker is not None:
                    tracked_boxes = self.tracker.update(raw_detections)
                else:
                    tracked_boxes = {i: detection[0] for i, detection in enumerate(raw_detections)}
                
                if enable_smoothing and self.smoother is not None:
                    final_boxes = self.smoother.smooth_boxes(tracked_boxes)
                else:
                    final_boxes = tracked_boxes
                
                boxes_to_blur = list(final_boxes.values())
                blurred_frame = self.apply_blur(frame, boxes_to_blur, blur_type, blur_strength)
                
                out.write(blurred_frame)
                
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress)
        
        finally:
            cap.release()
            out.release()


# Streamlit App
def main():
    st.set_page_config(
        page_title="YOLO Auto-Blur Video Processor",
        page_icon="🎥",
        layout="centered"
    )
    
    # Sidebar settings
    st.sidebar.header("Settings")
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="backend/runs/detect/blur_detection_model5/weights/best.pt",
        help="Path to your trained YOLO model"
    )
    
    blur_type = st.sidebar.selectbox(
        "Blur Type",
        options=['pixelate', 'gaussian', 'motion', 'black', 'white', 'mosaic'],
        index=0,
        help="Type of blur/censoring effect to apply"
    )
    
    blur_strength = st.sidebar.slider(
        "Blur Strength",
        min_value=5,
        max_value=50,
        value=15,
        help="Intensity of the blur effect"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    with st.sidebar.expander("Advanced Settings"):
        enable_tracking = st.checkbox("Enable Object Tracking", value=True)
        enable_smoothing = st.checkbox("Enable Temporal Smoothing", value=True)
        
        tracking_distance = st.slider(
            "Tracking Max Distance",
            min_value=10.0,
            max_value=200.0,
            value=100.0,
            help="Maximum distance for tracking objects between frames"
        )
            
        smoothing_factor = st.slider(
            "Smoothing Factor",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Temporal smoothing factor (higher = more responsive)"
        )
    
    # Main content
    st.title("YOLO Auto-Blur Video Processor")
    st.write("Upload a video to automatically detect and blur sensitive content.")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
    )
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / (1024*1024):.1f} MB")
        
        if st.button("Process Video", type="primary", use_container_width=True):
            try:
                if not os.path.exists(model_path):
                    st.error(f"Model not found at: {model_path}")
                    st.info("Please ensure you have a trained YOLO model and update the model path in the sidebar.")
                    return
                
                with st.spinner("Loading model..."):
                    blur_system = YOLOAutoBlur(model_path=model_path)
                    blur_system.load_model()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
                    temp_input.write(uploaded_file.read())
                    temp_input_path = temp_input.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                    temp_output_path = temp_output.name
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing... {progress*100:.1f}%")
                
                start_time = time.time()
                
                blur_system.detect_and_blur_video(
                    temp_input_path,
                    temp_output_path,
                    confidence_threshold=confidence_threshold,
                    blur_type=blur_type,
                    blur_strength=blur_strength,
                    enable_tracking=enable_tracking,
                    enable_smoothing=enable_smoothing,
                    tracking_max_distance=tracking_distance,
                    smoothing_factor=smoothing_factor,
                    progress_callback=update_progress
                )
                
                processing_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.text(f"Processing complete! ({processing_time:.1f}s)")
                
                if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                    with open(temp_output_path, 'rb') as f:
                        video_data = f.read()
                    
                    file_size_mb = len(video_data) / (1024 * 1024)
                    st.success(f"Video processed successfully! Output size: {file_size_mb:.1f} MB")
                    
                    st.download_button(
                        label="Download Processed Video",
                        data=video_data,
                        file_name=f"blurred_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    st.error("Processing failed - output file is empty or doesn't exist.")
                
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logger.error(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
