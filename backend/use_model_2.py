import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
import logging
from collections import deque
from scipy.spatial.distance import cdist
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


class YOLOAutoBlur:
    def __init__(self, model_path: str = None, data_yaml_path: str = "data.yaml"):
        """
        Initialize the YOLO Auto-Blur system
        
        Args:
            model_path: Path to trained YOLO model (if None, will use default or train new)
            data_yaml_path: Path to data.yaml configuration file
        """
        self.data_yaml_path = data_yaml_path
        self.model_path = model_path
        self.model = None
        self.class_names = self._load_class_names()
        
        # Video processing components
        self.tracker = None
        self.smoother = None
        
    def _load_class_names(self) -> dict:
        """Load class names from data.yaml"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            return data['names']
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            return {0: 'license_plate', 1: 'number', 2: 'identity_document'}
     
    def load_model(self, model_path: str = None):
        """Load a trained YOLO model"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("No model path provided. Please train a model first or provide a model path.")
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def apply_blur(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                   blur_type: str = "pixelate", blur_strength: int = 15) -> np.ndarray:
        """
        Apply blur/obscuring effects to specified regions in the image
        
        Args:
            image: Input image
            boxes: List of bounding boxes (x1, y1, x2, y2)
            blur_type: Type of blur - 'gaussian', 'motion', 'pixelate', 'black', 'white', 'mosaic'
            blur_strength: Strength of the blur effect
        
        Returns:
            Image with blurred regions
        """
        blurred_image = image.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract the region to blur
            region = image[y1:y2, x1:x2]
            
            if region.size > 0:  # Check if region is valid
                height, width = region.shape[:2]
                
                if blur_type == "gaussian":
                    # Ensure blur_strength is odd
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
                
                elif blur_type == "motion":
                    # Motion blur effect
                    kernel_size = max(blur_strength, 5)
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                    kernel = kernel / kernel_size
                    blurred_region = cv2.filter2D(region, -1, kernel)
                
                elif blur_type == "pixelate":
                    # Pixelate effect - more drastic
                    pixel_size = max(blur_strength // 3, 8)  # Larger pixels for more obvious effect
                    # Shrink image
                    small = cv2.resize(region, (width // pixel_size, height // pixel_size), 
                                     interpolation=cv2.INTER_LINEAR)
                    # Scale back up using nearest neighbor for blocky effect
                    blurred_region = cv2.resize(small, (width, height), 
                                              interpolation=cv2.INTER_NEAREST)
                
                elif blur_type == "black":
                    # Complete black box
                    blurred_region = np.zeros_like(region)
                
                elif blur_type == "white":
                    # Complete white box
                    blurred_region = np.ones_like(region) * 255
                
                elif blur_type == "mosaic":
                    # Mosaic effect - very obvious
                    tile_size = max(blur_strength // 2, 6)
                    blurred_region = region.copy()
                    for i in range(0, height, tile_size):
                        for j in range(0, width, tile_size):
                            # Get the average color of the tile
                            tile = region[i:i+tile_size, j:j+tile_size]
                            if tile.size > 0:
                                avg_color = np.mean(tile, axis=(0, 1))
                                blurred_region[i:i+tile_size, j:j+tile_size] = avg_color
                
                else:
                    # Default to heavy Gaussian blur
                    blur_strength = max(blur_strength, 21)
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
                
                # Replace the region in the image
                blurred_image[y1:y2, x1:x2] = blurred_region
        
        return blurred_image
    
    def detect_and_blur_image(self, image_path: str, output_path: str = None, 
                              confidence_threshold: float = 0.5, blur_type: str = "pixelate", 
                              blur_strength: int = 15) -> np.ndarray:
        """
        Detect ALL objects in an image and blur them (handles multiple instances)
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            confidence_threshold: Minimum confidence for detections
            blur_type: Type of blur ('gaussian', 'motion', 'pixelate', 'black', 'white', 'mosaic')
            blur_strength: Strength of the blur effect
        
        Returns:
            Processed image with blurred objects
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Run inference
        results = self.model(image, conf=confidence_threshold)
        
        # Extract ALL bounding boxes (YOLO detects multiple instances by default)
        boxes = []
        detection_count = {}  # Count detections per class
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Count detections per class
                    if class_name not in detection_count:
                        detection_count[class_name] = 0
                    detection_count[class_name] += 1
                    
                    logger.info(f"Detected {class_name} #{detection_count[class_name]} with confidence {confidence:.2f}")
                    boxes.append((x1, y1, x2, y2))
        
        # Log summary of detections
        total_detections = len(boxes)
        logger.info(f"Total detections: {total_detections}")
        for class_name, count in detection_count.items():
            logger.info(f"  - {class_name}: {count} instances")
        
        # Apply blur to ALL detected regions
        blurred_image = self.apply_blur(image, boxes, blur_type, blur_strength)
        
        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, blurred_image)
            logger.info(f"Processed image saved to: {output_path}")
        
        return blurred_image
    
    def detect_and_blur_video(self, video_path: str, output_path: str, 
                              confidence_threshold: float = 0.5, blur_type: str = "pixelate", 
                              blur_strength: int = 15, enable_tracking: bool = True,
                              enable_smoothing: bool = True, tracking_max_distance: float = 100.0,
                              smoothing_factor: float = 0.7):
        """
        Process a video and blur ALL detected objects with optional tracking and smoothing
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            confidence_threshold: Minimum confidence for detections
            blur_type: Type of blur ('gaussian', 'motion', 'pixelate', 'black', 'white', 'mosaic')
            blur_strength: Strength of the blur effect
            enable_tracking: Whether to use object tracking between frames
            enable_smoothing: Whether to apply temporal smoothing to bounding boxes
            tracking_max_distance: Maximum distance for tracking objects between frames
            smoothing_factor: Temporal smoothing factor (0-1, higher = more responsive)
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Initialize tracking and smoothing components
        if enable_tracking:
            self.tracker = BoundingBoxTracker(max_distance=tracking_max_distance)
            logger.info(f"Object tracking enabled (max_distance: {tracking_max_distance})")
        
        if enable_smoothing:
            self.smoother = TemporalSmoother(smoothing_factor=smoothing_factor)
            logger.info(f"Temporal smoothing enabled (smoothing_factor: {smoothing_factor})")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        logger.info(f"Using blur type: {blur_type}")
        
        frame_count = 0
        total_detections_video = 0
        total_tracks = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference on frame
                results = self.model(frame, conf=confidence_threshold)
                
                # Extract ALL bounding boxes and class IDs
                raw_detections = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            raw_detections.append(((x1, y1, x2, y2), class_id))
                
                total_detections_video += len(raw_detections)
                
                # Apply tracking if enabled
                if enable_tracking and self.tracker is not None:
                    tracked_boxes = self.tracker.update(raw_detections)
                    total_tracks = max(total_tracks, len(tracked_boxes))
                else:
                    # Use raw detections without tracking
                    tracked_boxes = {i: detection[0] for i, detection in enumerate(raw_detections)}
                
                # Apply temporal smoothing if enabled
                if enable_smoothing and self.smoother is not None:
                    final_boxes = self.smoother.smooth_boxes(tracked_boxes)
                else:
                    final_boxes = tracked_boxes
                
                # Convert to list format for blur application
                boxes_to_blur = list(final_boxes.values())
                
                # Apply blur to ALL detected/tracked regions
                blurred_frame = self.apply_blur(frame, boxes_to_blur, blur_type, blur_strength)
                
                # Write frame
                out.write(blurred_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_detections = total_detections_video / frame_count
                    active_tracks = len(final_boxes)
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                               f"Avg detections/frame: {avg_detections:.1f}, Active tracks: {active_tracks}")
        
        finally:
            # Clean up
            cap.release()
            out.release()
            
        logger.info(f"Video processing complete. Output saved to: {output_path}")
        logger.info(f"Total detections across all frames: {total_detections_video}")
        if enable_tracking:
            logger.info(f"Maximum concurrent tracks: {total_tracks}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='YOLO Auto-Blur Detection System with Tracking')
    parser.add_argument('--mode', choices=['train', 'image', 'video'], required=True,
                       help='Operation mode: train model, process image, or process video')
    parser.add_argument('--input', type=str, help='Input file path (for image/video modes)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--model', type=str, help='Path to trained model (for image/video modes)')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--blur-type', type=str, default='pixelate', 
                       choices=['gaussian', 'motion', 'pixelate', 'black', 'white', 'mosaic'],
                       help='Type of blur/obscuring effect')
    parser.add_argument('--blur', type=int, default=15, help='Blur strength')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (train mode)')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    
    # Video processing arguments
    parser.add_argument('--no-tracking', action='store_true', help='Disable object tracking')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable temporal smoothing')
    parser.add_argument('--tracking-distance', type=float, default=100.0, 
                       help='Maximum distance for tracking objects between frames')
    parser.add_argument('--smoothing-factor', type=float, default=0.7,
                       help='Temporal smoothing factor (0-1, higher = more responsive)')
    
    args = parser.parse_args()
    
    # Initialize the auto-blur system
    blur_system = YOLOAutoBlur(model_path=args.model, data_yaml_path=args.data)
    
    if args.mode == 'image':
        if not args.input:
            raise ValueError("Input image path required for image mode")
        
        # Load model and process image
        blur_system.load_model()
        output_path = args.output or f"blurred_{Path(args.input).name}"
        blur_system.detect_and_blur_image(
            args.input, output_path, args.confidence, args.blur_type, args.blur
        )
        
    elif args.mode == 'video':
        if not args.input or not args.output:
            raise ValueError("Both input and output paths required for video mode")
        
        # Load model and process video
        blur_system.load_model()
        blur_system.detect_and_blur_video(
            args.input, args.output, args.confidence, args.blur_type, args.blur,
            enable_tracking=not args.no_tracking,
            enable_smoothing=not args.no_smoothing,
            tracking_max_distance=args.tracking_distance,
            smoothing_factor=args.smoothing_factor
        )


def example_image_processing(input_image):
    """Example: Process a single image"""
    print("=== Image Processing Example ===")
    
    # Initialize with trained model
    blur_system = YOLOAutoBlur(model_path="runs/detect/blur_detection_model5/weights/best.pt")
    blur_system.load_model()
    
    # Process image
    output_image = f"blurred_{input_image}"
    
    if os.path.exists(input_image):
        processed_img = blur_system.detect_and_blur_image(
            input_image, output_image, confidence_threshold=0.25, 
            blur_type="black", blur_strength=21
        )
        print(f"Image processed and saved to {output_image}")
    else:
        print(f"Test image {input_image} not found")


def example_video_processing(input_video):
    """Example: Process a video with tracking and smoothing"""
    print("=== Video Processing Example with Tracking & Smoothing ===")
    
    # Initialize with trained model
    blur_system = YOLOAutoBlur(model_path="runs/detect/blur_detection_model5/weights/best.pt")
    blur_system.load_model()
    
    # Process video with enhanced features
    output_video = f"blurred_tracked_{input_video}"
    
    if os.path.exists(input_video):
        blur_system.detect_and_blur_video(
            input_video, output_video, 
            confidence_threshold=0.5, 
            blur_type="mosiac", 
            blur_strength=21,
            enable_tracking=True,
            enable_smoothing=True,
            tracking_max_distance=70.0,
            smoothing_factor=0.6
        )
        print(f"Video processed and saved to {output_video}")
    else:
        print(f"Test video {input_video} not found")


if __name__ == "__main__":
    # Uncomment the examples you want to run:
    
    # For testing on images:
    #example_image_processing("test_lp_1.jpg")
    
    # For testing on videos with enhanced processing:
    example_video_processing("cars_1.mp4")
    
    # For command line usage:
    #main()