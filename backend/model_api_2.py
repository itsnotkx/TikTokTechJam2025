import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import logging
import uuid
from collections import deque, defaultdict
from scipy.spatial.distance import cdist
import math
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Optional, Tuple
import json
import shutil
from pathlib import Path
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="PII Detection and Censoring API", version="2.0.0")

# Configuration
UPLOAD_DIR = "uploaded_videos"
PROCESSED_DIR = "processed_videos"
MODEL_PATH = "runs/detect/blur_detection_model5/weights/best.pt"
DATA_YAML_PATH = "data.yaml"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


class BoundingBoxTracker:
    """Simple tracker for bounding boxes across frames using distance-based matching"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        """
        Initialize the tracker
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for matching boxes between frames
        """
        self.next_id = 0
        self.tracks = {}  # track_id: {'box': (x1,y1,x2,y2), 'disappeared': int, 'class_id': int}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def _get_center(self, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, box1: Tuple[float, float, float, float], 
                           box2: Tuple[float, float, float, float]) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = self._get_center(box1)
        center2 = self._get_center(box2)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections: List[Tuple[Tuple[float, float, float, float], int]]) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of (box, class_id) tuples
            
        Returns:
            Dictionary of track_id: box mappings
        """
        if len(detections) == 0:
            # No detections - increment disappeared counter for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return {track_id: track['box'] for track_id, track in self.tracks.items()}
        
        if len(self.tracks) == 0:
            # No existing tracks - create new tracks for all detections
            for box, class_id in detections:
                self.tracks[self.next_id] = {
                    'box': box,
                    'disappeared': 0,
                    'class_id': class_id
                }
                self.next_id += 1
            return {track_id: track['box'] for track_id, track in self.tracks.items()}
        
        # Calculate distance matrix between existing tracks and new detections
        track_centers = [self._get_center(track['box']) for track in self.tracks.values()]
        detection_centers = [self._get_center(box) for box, _ in detections]
        
        if len(track_centers) > 0 and len(detection_centers) > 0:
            distance_matrix = cdist(track_centers, detection_centers)
            
            # Find best matches using simple greedy assignment
            used_detection_indices = set()
            used_track_indices = set()
            track_ids = list(self.tracks.keys())
            
            # Sort by distance and assign matches
            matches = []
            for i in range(len(track_ids)):
                for j in range(len(detections)):
                    if i not in used_track_indices and j not in used_detection_indices:
                        if distance_matrix[i, j] < self.max_distance:
                            matches.append((i, j, distance_matrix[i, j]))
            
            matches.sort(key=lambda x: x[2])  # Sort by distance
            
            # Apply matches
            for track_idx, detection_idx, distance in matches:
                if track_idx not in used_track_indices and detection_idx not in used_detection_indices:
                    track_id = track_ids[track_idx]
                    box, class_id = detections[detection_idx]
                    
                    self.tracks[track_id]['box'] = box
                    self.tracks[track_id]['disappeared'] = 0
                    self.tracks[track_id]['class_id'] = class_id
                    
                    used_track_indices.add(track_idx)
                    used_detection_indices.add(detection_idx)
            
            # Handle unmatched tracks
            for i, track_id in enumerate(track_ids):
                if i not in used_track_indices:
                    self.tracks[track_id]['disappeared'] += 1
                    if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                        del self.tracks[track_id]
            
            # Handle unmatched detections (create new tracks)
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
        """
        Initialize temporal smoother
        
        Args:
            window_size: Number of frames to consider for smoothing
            smoothing_factor: Weight for current frame vs history (0-1)
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.history = {}  # track_id: deque of boxes
    
    def smooth_boxes(self, tracked_boxes: Dict[int, Tuple[float, float, float, float]]) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Apply temporal smoothing to tracked boxes
        
        Args:
            tracked_boxes: Dictionary of track_id: box mappings
            
        Returns:
            Smoothed boxes
        """
        smoothed_boxes = {}
        
        for track_id, box in tracked_boxes.items():
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.window_size)
            
            self.history[track_id].append(box)
            
            if len(self.history[track_id]) == 1:
                # First frame for this track
                smoothed_boxes[track_id] = box
            else:
                # Apply exponential moving average smoothing
                prev_box = list(self.history[track_id])[-2]  # Previous box
                current_box = box
                
                smoothed_box = tuple(
                    self.smoothing_factor * current + (1 - self.smoothing_factor) * prev
                    for current, prev in zip(current_box, prev_box)
                )
                smoothed_boxes[track_id] = smoothed_box
        
        # Clean up history for tracks that are no longer active
        active_track_ids = set(tracked_boxes.keys())
        inactive_track_ids = set(self.history.keys()) - active_track_ids
        for track_id in inactive_track_ids:
            del self.history[track_id]
        
        return smoothed_boxes


class TimestampedTracker:
    """Extended tracker that maintains timing information for API compatibility"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        self.tracker = BoundingBoxTracker(max_disappeared, max_distance)
        self.smoother = TemporalSmoother()
        self.track_timestamps = {}  # track_id: {'start_time': float, 'end_time': float}
        self.track_coords_history = {}  # track_id: list of normalized coords
    
    def update_with_timing(self, detections: List[Tuple[Tuple[float, float, float, float], int]], 
                          frame_time: float, frame_width: int, frame_height: int):
        """Update tracks with timing information"""
        # Get tracked boxes
        tracked_boxes = self.tracker.update(detections)
        
        # Apply smoothing
        smoothed_boxes = self.smoother.smooth_boxes(tracked_boxes)
        
        # Update timing information
        active_track_ids = set(smoothed_boxes.keys())
        
        # Update existing tracks
        for track_id, box in smoothed_boxes.items():
            if track_id not in self.track_timestamps:
                # New track
                self.track_timestamps[track_id] = {
                    'start_time': frame_time,
                    'end_time': frame_time
                }
                self.track_coords_history[track_id] = []
            else:
                # Update end time for active track
                self.track_timestamps[track_id]['end_time'] = frame_time
            
            # Store normalized coordinates
            normalized_coords = self.normalize_coords(box, frame_width, frame_height)
            self.track_coords_history[track_id].append(normalized_coords)
        
        # Update end times for disappeared tracks
        disappeared_tracks = set(self.track_timestamps.keys()) - active_track_ids
        for track_id in disappeared_tracks:
            # Don't update end time if track just disappeared, keep last known good time
            pass
        
        return smoothed_boxes
    
    def normalize_coords(self, box, frame_width, frame_height):
        """Normalize coordinates to 0-1 range"""
        x1, y1, x2, y2 = box
        return [float(x1/frame_width), float(y1/frame_height), float(x2/frame_width), float(y2/frame_height)]
    
    def get_pii_objects(self):
        """Get PII objects in API format"""
        pii_objects = []
        for track_id, timestamps in self.track_timestamps.items():
            if track_id in self.track_coords_history and len(self.track_coords_history[track_id]) > 0:
                # Use average coordinates from history
                avg_coords = np.mean(self.track_coords_history[track_id], axis=0).tolist()
                
                pii_obj = {
                    "start_time": float(timestamps['start_time']),
                    "end_time": float(timestamps['end_time']),
                    "coords": avg_coords,
                    "track_id": int(track_id)
                }
                pii_objects.append(pii_obj)
        
        return pii_objects


def load_class_names(data_yaml_path=DATA_YAML_PATH):
    """Load class names from data.yaml"""
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        return {0: 'license_plate', 1: 'number', 2: 'identity_document'}


def detect_pii_in_video(video_path, video_id):
    """
    Detect PII objects in video using robust CLI logic
    """
    model = YOLO(MODEL_PATH)
    class_names = load_class_names()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video from {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing video {video_id}: {total_frames} frames at {fps} FPS")
    
    # Initialize enhanced tracker
    timestamped_tracker = TimestampedTracker(max_distance=70.0)
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time = frame_count / fps
            
            # Run inference with confidence threshold matching CLI
            results = model(frame, conf=0.5)
            
            # Extract ALL bounding boxes and class IDs (like CLI version)
            raw_detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Log detections for debugging
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        logger.debug(f"Frame {frame_count}: Detected {class_name} with confidence {confidence:.2f}")
                        
                        raw_detections.append(((x1, y1, x2, y2), class_id))
            
            total_detections += len(raw_detections)
            
            # Update tracking with timing
            smoothed_boxes = timestamped_tracker.update_with_timing(
                raw_detections, frame_time, frame_width, frame_height
            )
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                active_tracks = len(smoothed_boxes)
                avg_detections = total_detections / frame_count if frame_count > 0 else 0
                logger.info(f"Detection progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                           f"Active tracks: {active_tracks}, Avg detections/frame: {avg_detections:.1f}")
    
    finally:
        cap.release()
    
    # Get PII objects
    pii_objects = timestamped_tracker.get_pii_objects()
    
    logger.info(f"Detected {len(pii_objects)} PII objects in video {video_id}")
    logger.info(f"Total raw detections: {total_detections}")
    
    return pii_objects, float(fps), int(frame_width), int(frame_height)


def apply_mosaic_blur(image, boxes, blur_strength=15):
    """Apply mosaic blur to specified regions (same as original API)"""
    blurred_image = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        if x1 >= x2 or y1 >= y2:
            continue
            
        region = image[y1:y2, x1:x2]
        
        if region.size > 0:
            height, width = region.shape[:2]
            tile_size = max(blur_strength // 2, 6)
            blurred_region = region.copy()
            
            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    tile = region[i:i+tile_size, j:j+tile_size]
                    if tile.size > 0:
                        avg_color = np.mean(tile, axis=(0, 1))
                        blurred_region[i:i+tile_size, j:j+tile_size] = avg_color
            
            blurred_image[y1:y2, x1:x2] = blurred_region
    
    return blurred_image


def denormalize_coords(coords, frame_width, frame_height):
    """Convert normalized coordinates back to pixel coordinates"""
    x1, y1, x2, y2 = coords
    return [float(x1*frame_width), float(y1*frame_height), float(x2*frame_width), float(y2*frame_height)]


def censor_pii_in_video(video_path, output_path, pii_objects_to_censor, fps, frame_width, frame_height):
    """
    Censor specific PII objects in video based on timing and coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video from {video_path}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Find PII objects that should be censored at this time
            boxes_to_blur = []
            for pii_obj in pii_objects_to_censor:
                if pii_obj["start_time"] <= current_time <= pii_obj["end_time"]:
                    # Convert normalized coordinates back to pixel coordinates
                    coords = denormalize_coords(pii_obj["coords"], frame_width, frame_height)
                    boxes_to_blur.append(coords)
            
            # Apply mosaic blur to active regions
            if boxes_to_blur:
                frame = apply_mosaic_blur(frame, boxes_to_blur, blur_strength=21)
            
            out.write(frame)
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                logger.info(f"Censoring progress: frame {frame_count}")
    
    finally:
        cap.release()
        out.release()


@app.post("/detect-pii")
async def detect_pii(file: UploadFile = File(...)):
    """
    Upload video, detect PII objects using robust CLI logic
    """
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Save uploaded video
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded video saved as {video_id}")
        
        # Process video to detect PII using enhanced logic
        pii_objects, fps, frame_width, frame_height = detect_pii_in_video(video_path, video_id)
        
        # Store video metadata for later use
        metadata = {
            "fps": float(fps),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height)
        }
        metadata_path = os.path.join(UPLOAD_DIR, f"{video_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        return {
            "video_id": video_id,
            "pii_objects": pii_objects
        }
        
    except Exception as e:
        logger.error(f"Error in detect_pii: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/censor-pii")
async def censor_pii(request: Request):
    """
    Censor specified PII objects in the video
    """
    try:
        # Parse JSON request
        data = await request.json()
        video_id = data.get("video_id")
        pii_objects = data.get("pii_objects", [])
        
        if not video_id:
            raise HTTPException(status_code=400, detail="video_id is required")
        
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
        metadata_path = os.path.join(UPLOAD_DIR, f"{video_id}_metadata.json")
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Video metadata not found")
        
        # Load video metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        fps = float(metadata["fps"])
        frame_width = int(metadata["frame_width"])
        frame_height = int(metadata["frame_height"])
        
        # Create output path
        blurred_video_id = f"blurred_{video_id}"
        output_path = os.path.join(PROCESSED_DIR, f"{blurred_video_id}.mp4")
        
        logger.info(f"Censoring {len(pii_objects)} PII objects in video {video_id}")
        
        # Process video with censoring
        censor_pii_in_video(video_path, output_path, pii_objects, fps, frame_width, frame_height)
        
        return {
            "success": True,
            "message": f"Successfully censored {len(pii_objects)} PII objects",
            "blurred_video_id": blurred_video_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in censor_pii: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-video/{video_id}")
async def download_video(video_id: str):
    """
    Download the censored video
    """
    try:
        # Look for blurred video
        video_path = os.path.join(PROCESSED_DIR, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Censored video not found")
        
        return FileResponse(
            path=video_path,
            media_type='video/mp4',
            filename=f"{video_id}.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PII Detection and Censoring API v2.0 is running"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PII Detection and Censoring API v2.0 - Enhanced with CLI Logic",
        "version": "2.0.0",
        "features": ["Object Tracking", "Temporal Smoothing", "Robust Detection"],
        "endpoints": {
            "/detect-pii": "POST - Upload video and detect PII objects",
            "/censor-pii": "POST - Censor specified PII objects in video", 
            "/download-video/{video_id}": "GET - Download censored video",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)