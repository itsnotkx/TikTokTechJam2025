import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOAutoBlur:
    def __init__(self, model_path: str = None, data_yaml_path: str = r"merged_yolo/data.yaml"):
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
        
    def _load_class_names(self) -> dict:
        """Load class names from data.yaml"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            return data['names']
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            return {0: 'license_plate', 1: 'number', 2: 'identity_document'}
    
    def train_model(self, epochs: int = 100, img_size: int = 640, batch_size: int = 16):
        """
        Train a new YOLO model
        
        Args:
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size for training
        """
        logger.info("Starting model training...")
        
        # Initialize YOLO model (you can also use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        model = YOLO('yolov8n.pt')  # Start with pretrained weights
        
        # Train the model
        results = model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=-1,
            name='blur_detection_model',
            save=True,
            verbose=True,
            device = 0,
        )
        
        # Save the best model path
        self.model_path = model.trainer.best
        logger.info(f"Training completed. Best model saved at: {self.model_path}")
        
        return results
    
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
                   blur_strength: int =3005) -> np.ndarray:
        """
        Apply blur to specified regions in the image
        
        Args:
            image: Input image
            boxes: List of bounding boxes (x1, y1, x2, y2)
            blur_strength: Strength of the blur effect (odd number)
        
        Returns:
            Image with blurred regions
        """
        # Ensure blur_strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        blurred_image = image.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract the region to blur
            region = image[y1:y2, x1:x2]
            
            if region.size > 0:  # Check if region is valid
                # Apply Gaussian blur
                blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
                
                # Replace the region in the image
                blurred_image[y1:y2, x1:x2] = blurred_region
        
        return blurred_image
    
    def detect_and_blur_image(self, image_path: str, output_path: str = None, 
                              confidence_threshold: float = 0.5, blur_strength: int = 15) -> np.ndarray:
        """
        Detect objects in an image and blur them
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            confidence_threshold: Minimum confidence for detections
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
        
        # Extract bounding boxes
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    logger.info(f"Detected {self.class_names[class_id]} with confidence {confidence:.2f}")
                    boxes.append((x1, y1, x2, y2))
        
        # Apply blur to detected regions
        blurred_image = self.apply_blur(image, boxes, blur_strength)
        
        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, blurred_image)
            logger.info(f"Processed image saved to: {output_path}")
        
        return blurred_image
    
    def detect_and_blur_video(self, video_path: str, output_path: str, 
                              confidence_threshold: float = 0.5, blur_strength: int = 15):
        """
        Process a video and blur detected objects
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            confidence_threshold: Minimum confidence for detections
            blur_strength: Strength of the blur effect
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load a model first.")
        
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
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference on frame
                results = self.model(frame, conf=confidence_threshold)
                
                # Extract bounding boxes
                boxes = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append((x1, y1, x2, y2))
                
                # Apply blur to detected regions
                blurred_frame = self.apply_blur(frame, boxes, blur_strength)
                
                # Write frame
                out.write(blurred_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            # Clean up
            cap.release()
            out.release()
            
        logger.info(f"Video processing complete. Output saved to: {output_path}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='YOLO Auto-Blur Detection System')
    parser.add_argument('--mode', choices=['train', 'image', 'video'], required=True,
                       help='Operation mode: train model, process image, or process video')
    parser.add_argument('--input', type=str, help='Input file path (for image/video modes)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--model', type=str, help='Path to trained model (for image/video modes)')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--blur', type=int, default=15, help='Blur strength (odd number)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (train mode)')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Initialize the auto-blur system
    blur_system = YOLOAutoBlur(model_path=args.model, data_yaml_path=args.data)
    
    if args.mode == 'train':
        # Train the model
        blur_system.train_model(epochs=args.epochs, img_size=args.img_size, batch_size=args.batch_size)
        
    elif args.mode == 'image':
        if not args.input:
            raise ValueError("Input image path required for image mode")
        
        # Load model and process image
        blur_system.load_model()
        output_path = args.output or f"blurred_{Path(args.input).name}"
        blur_system.detect_and_blur_image(
            args.input, output_path, args.confidence, args.blur
        )
        
    elif args.mode == 'video':
        if not args.input or not args.output:
            raise ValueError("Both input and output paths required for video mode")
        
        # Load model and process video
        blur_system.load_model()
        blur_system.detect_and_blur_video(
            args.input, args.output, args.confidence, args.blur
        )


# Example usage functions
def example_training():
    """Example: Train a new model"""
    print("=== Training Example ===")
    blur_system = YOLOAutoBlur(data_yaml_path="merged_yolo/data.yaml")
    
    # Train with custom parameters
    results = blur_system.train_model(epochs=30, img_size=640, batch_size=8)
    print(f"Training completed!")


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
            input_image, output_image, confidence_threshold=0.5, blur_strength=21
        )
        print(f"Image processed and saved to {output_image}")
    else:
        print(f"Test image {input_image} not found")


def example_video_processing(input_video):
    """Example: Process a video"""
    print("=== Video Processing Example ===")
    
    # Initialize with trained model
    blur_system = YOLOAutoBlur(model_path="runs/detect/blur_detection_model5/weights/best.pt")
    blur_system.load_model()
    
    # Process video
    output_video = f"blurred_{input_video}"
    
    if os.path.exists(input_video):
        blur_system.detect_and_blur_video(
            input_video, output_video, confidence_threshold=0.5, blur_strength=21
        )
        print(f"Video processed and saved to {output_video}")
    else:
        print(f"Test video {input_video} not found")


if __name__ == "__main__":
    # Uncomment the examples you want to run:
    
    # For training:
    #example_training()
    
    # For testing on images:
    example_image_processing("test_lp_2.jpg")
    
    # For testing on videos:
    #example_video_processing("test_block_num_2.mp4")
    
    # For command line usage:
    main()