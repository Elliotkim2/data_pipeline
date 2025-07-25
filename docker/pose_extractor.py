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
            2: 15,   # right eye
            5: 16,   # left eye
            7: 17,   # right ear
            8: 18,   # left ear
            11: 1,   # Use left shoulder for neck approximation
            12: 2,   # right shoulder  
            13: 6,   # left elbow
            14: 3,   # right elbow
            15: 7,   # left wrist
            16: 4,   # right wrist
            23: 12,  # left hip
            24: 8,   # mid hip
            25: 13,  # left knee
            26: 10,  # right knee
            27: 14,  # left ankle
            28: 11,  # right ankle
            29: 21,  # left heel
            30: 24,  # right heel
            31: 19,  # left big toe
            32: 22,  # right big toe
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
        
        try:
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
            
            # Convert to OpenPose format with error checking
            h, w, _ = frame.shape
            openpose_data = {"people": [{"pose_keypoints_2d": []}]}
            keypoints = []
            
            # Map MediaPipe landmarks to OpenPose format
            for op_idx in range(25):
                mp_idx = self.mp_to_op.get(op_idx, 0)
                
                # Validate landmark index
                if mp_idx >= len(results.pose_landmarks.landmark):
                    logger.warning(f"Invalid landmark index {mp_idx}, using 0")
                    mp_idx = 0
                
                landmark = results.pose_landmarks.landmark[mp_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * w
                y = landmark.y * h
                confidence = landmark.visibility
                
                keypoints.extend([float(x), float(y), float(confidence)])
            
            openpose_data["people"][0]["pose_keypoints_2d"] = keypoints
            return openpose_data
            
        except Exception as e:
            logger.error(f"Error extracting pose: {e}")
            return None
    
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

    def extract_poses_from_frame_list(self, frame_files, poses_dir):
        """
        Extract poses from frame files and save to directory
        
        Args:
            frame_files: List of (frame_path, frame_filename) tuples
            poses_dir: Directory to save pose JSON files
        
        Returns:
            List of (pose_path, pose_filename) tuples
        """
        import os
        os.makedirs(poses_dir, exist_ok=True)
        pose_files = []
        
        for i, (frame_path, frame_filename) in enumerate(frame_files):
            if not os.path.exists(frame_path):
                logger.warning(f"Frame file not found: {frame_path}")
                continue
                
            pose_data = self.extract_pose_from_frame(frame_path)
            
            if pose_data:
                # Create pose filename
                pose_filename = frame_filename.replace('frame_', 'pose_').replace('.jpg', '.json')
                pose_path = os.path.join(poses_dir, pose_filename)
                
                # Save pose data
                with open(pose_path, 'w') as f:
                    json.dump(pose_data, f)
                
                pose_files.append((pose_path, pose_filename))
                logger.info(f"Extracted pose {i+1}/{len(frame_files)}: {pose_filename}")
        
        return pose_files