# annotation-pipeline/lambda/pose_extractor.py
import cv2
import mediapipe as mp
import numpy as np
import json
import logging

logger = logging.getLogger()

class PoseExtractor:
    def __init__(self):
        """Initialize MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.pose = None
        self._initialized = False
        
        # MediaPipe to OpenPose mapping (25 keypoints)
        self.mp_to_op = {
            0: 0,    # nose
            11: 1,   # neck (left shoulder)
            12: 2,   # right shoulder  
            14: 3,   # right elbow
            16: 4,   # right wrist
            11: 5,   # left shoulder
            13: 6,   # left elbow
            15: 7,   # left wrist
            24: 8,   # mid hip
            24: 9,   # right hip (using mid hip)
            26: 10,  # right knee
            28: 11,  # right ankle
            23: 12,  # left hip
            25: 13,  # left knee
            27: 14,  # left ankle
            2: 15,   # right eye
            5: 16,   # left eye
            7: 17,   # right ear
            8: 18,   # left ear
            31: 19,  # left big toe
            31: 20,  # left small toe
            29: 21,  # left heel
            32: 22,  # right big toe
            32: 23,  # right small toe
            30: 24   # right heel
        }
    
    def _lazy_init(self):
        """Lazy initialization of MediaPipe"""
        if not self._initialized:
            logger.info("Initializing MediaPipe Pose...")
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,  # For individual frames
                model_complexity=1,      # Balance speed/accuracy
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self._initialized = True
    
    def extract_pose_from_frame(self, frame_path_or_array):
        """
        Extract pose from a single frame
        
        Args:
            frame_path_or_array: Either file path or numpy array
            
        Returns:
            dict: OpenPose format pose data
        """
        self._lazy_init()
        
        # Load frame if path provided
        if isinstance(frame_path_or_array, str):
            frame = cv2.imread(frame_path_or_array)
            if frame is None:
                logger.error(f"Could not load frame: {frame_path_or_array}")
                return None
        else:
            frame = frame_path_or_array
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            logger.warning("No pose detected in frame")
            return None
        
        # Convert to OpenPose format
        h, w, _ = frame.shape
        openpose_data = {"people": [{"pose_keypoints_2d": []}]}
        keypoints = []
        
        # Map MediaPipe landmarks to OpenPose format
        for op_idx in range(25):
            mp_idx = self.mp_to_op.get(op_idx, 0)
            landmark = results.pose_landmarks.landmark[mp_idx]
            
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * w
            y = landmark.y * h
            confidence = landmark.visibility
            
            keypoints.extend([float(x), float(y), float(confidence)])
        
        openpose_data["people"][0]["pose_keypoints_2d"] = keypoints
        
        return openpose_data
    
    def extract_poses_from_frames(self, frame_paths, max_frames=10):
        """
        Extract poses from multiple frames
        
        Args:
            frame_paths: List of frame file paths
            max_frames: Maximum number of frames to process
            
        Returns:
            list: List of (frame_path, pose_data) tuples
        """
        poses = []
        
        for i, frame_path in enumerate(frame_paths[:max_frames]):
            pose_data = self.extract_pose_from_frame(frame_path)
            if pose_data:
                poses.append((frame_path, pose_data))
            
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1} frames for pose extraction")
        
        return poses
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if self.pose:
            self.pose.close()