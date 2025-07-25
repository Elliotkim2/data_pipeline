import os
import json
import logging
import re
from datetime import datetime
import whisper
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import pandas as pd
from pose_extractor import PoseExtractor
import boto3
s3 = boto3.client('s3')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global Whisper models (reused across invocations)
whisper_models = {}

def get_whisper_model(model_size="base"):
    """Get or load Whisper model with caching"""
    global whisper_models
    
    if model_size not in whisper_models:
        logger.info(f"Loading Whisper {model_size} model...")
        # Force all cache directories to /tmp
        import os
        os.environ['TORCH_HOME'] = '/tmp'
        os.environ['HF_HOME'] = '/tmp'
        os.environ['TRANSFORMERS_CACHE'] = '/tmp'
        os.environ['XDG_CACHE_HOME'] = '/tmp'
        
        # Load model with explicit download root
        whisper_models[model_size] = whisper.load_model(model_size, download_root="/tmp/whisper-cache")
        logger.info(f"Whisper {model_size} model loaded successfully")
    
    return whisper_models[model_size]


def extract_audio_from_video(video_path):
    """Extract audio from video and save as temporary WAV file"""
    temp_audio_path = "/tmp/temp_audio.wav"
    
    try:
        logger.info("Extracting audio from video...")
        video = VideoFileClip(video_path)
        duration = video.duration
        
        if video.audio is not None:
            video.audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
            video.close()
            return temp_audio_path, duration
        else:
            video.close()
            return None, duration
            
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None, 0

def transcribe_with_whisper(audio_path, model_size="base"):
    """Transcribe audio using Whisper"""
    try:
        model = get_whisper_model(model_size)
        logger.info("Transcribing audio with Whisper...")
        
        result = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            verbose=False
        )
        
        return result
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return {"text": "", "segments": []}
    
def extract_poses_from_frame_list(self, frame_files, poses_dir):
    """
    Extract poses from a list of frame files and save to poses directory
    
    Args:
        frame_files: List of tuples (frame_path, frame_filename) from process_video
        poses_dir: Directory to save pose JSON files
        
    Returns:
        list: List of saved pose file paths
    """
    os.makedirs(poses_dir, exist_ok=True)
    pose_files = []
    
    for i, (frame_path, frame_filename) in enumerate(frame_files):
        if not os.path.exists(frame_path):
            logger.warning(f"Frame file not found: {frame_path}")
            continue
            
        pose_data = self.extract_pose_from_frame(frame_path)
        
        if pose_data:
            # Create pose filename based on frame filename
            # frame_001_at_1.23s_john_happy.jpg -> pose_001_at_1.23s_john_happy.json
            pose_filename = frame_filename.replace('frame_', 'pose_').replace('.jpg', '.json')
            pose_path = os.path.join(poses_dir, pose_filename)
            
            # Save pose data
            with open(pose_path, 'w') as f:
                json.dump(pose_data, f)
            
            pose_files.append((pose_path, pose_filename))
            logger.info(f"Extracted pose {i+1}/{len(frame_files)}: {pose_filename}")
    
    return pose_files

def __del__(self):
    """Cleanup MediaPipe resources"""
    if self.pose:
        self.pose.close()

def process_video(input_path, output_path):
    """Main processing function for Lambda with Whisper"""
    try:
        logger.info(f"Processing video: {input_path}")
        
        # Initialize emotion keywords
        emotion_keywords = {
            'happy': ['happy', 'joyful', 'excited', 'cheerful', 'delighted', 'glad', 'pleased', 'thrilled'],
            'sad': ['sad', 'unhappy', 'depressed', 'miserable', 'crying', 'upset', 'disappointed'],
            'surprised': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'startled'],
            'neutral': ['neutral', 'calm', 'normal', 'fine', 'okay', 'alright', 'composed'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated']
        }
        
        # Create reverse mapping
        keyword_to_emotion = {}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                keyword_to_emotion[keyword.lower()] = emotion
        
        # Extract audio and get duration
        temp_audio, duration = extract_audio_from_video(input_path)
        
        # Transcribe if audio exists
        if temp_audio and os.path.exists(temp_audio):
            # Use Whisper for transcription
            whisper_result = transcribe_with_whisper(temp_audio, model_size="base")
            transcription = whisper_result.get('text', '')
            segments = whisper_result.get('segments', [])
            
            # Clean up audio file
            os.remove(temp_audio)
            
            logger.info(f"Transcription: {transcription}")
        else:
            transcription = ""
            segments = []
        
        # Detect annotation patterns
        annotation_patterns = [
            r"this is (?:the|a) video (?:for|of) (\w+) being (\w+)",
            r"(\w+) is (\w+) in this video",
            r"video of (\w+) feeling (\w+)",
            r"(\w+) looks (\w+)",
            r"showing (\w+) who is (\w+)",
            r"(\w+\.?\s*\w+) being (\w+)"  # Handles "Mrs. Elliot being happy"
        ]
        
        annotations = []
        text_lower = transcription.lower()
        
        for pattern in annotation_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                person_name = match.group(1)
                emotion_word = match.group(2)
                if emotion_word in keyword_to_emotion:
                    emotion = keyword_to_emotion[emotion_word]
                    annotations.append({
                        'person': person_name,
                        'emotion': emotion,
                        'text': match.group(0)
                    })
        
        # Extract emotion keywords
        emotions_found = []
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in keyword_to_emotion:
                emotion = keyword_to_emotion[word]
                emotions_found.append(f"{word} ({emotion})")
        
        # Determine dominant emotion
        emotion_counts = {}
        for word in words:
            if word in keyword_to_emotion:
                emotion = keyword_to_emotion[word]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Extract frames when emotions are found
        frames_extracted = 0
        frame_files = []
        
        if emotions_found:
            logger.info("Extracting frames for detected emotions...")
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create frames directory
            frames_dir = "/tmp/frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extract frames based on segments if available
            if segments and annotations:
                # Extract frames at annotation timestamps
                for annotation in annotations:
                    # Find segment containing annotation
                    for segment in segments:
                        if annotation['text'] in segment['text'].lower():
                            timestamp = segment['start']
                            frame_num = int(timestamp * fps)
                            
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                            ret, frame = cap.read()
                            if ret:
                                frame_filename = f"frame_{frames_extracted:03d}_at_{timestamp:.2f}s_{annotation['person']}_{annotation['emotion']}.jpg"
                                frame_path = os.path.join(frames_dir, frame_filename)
                                cv2.imwrite(frame_path, frame)
                                frame_files.append((frame_path, frame_filename))
                                frames_extracted += 1
            
            # Also extract regular interval frames
            interval = max(1, total_frames // 12)
            for i in range(0, min(total_frames, 12 * interval), interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame_filename = f"frame_{frames_extracted:03d}_at_pos_{i}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_files.append((frame_path, frame_filename))
                    frames_extracted += 1
            
            cap.release()
        #create json of body poses
        poses_extracted = 0
        pose_files = []
        
        if frame_files and dominant_emotion != 'neutral':
            logger.info(f"Extracting poses from {len(frame_files)} frames...")
            
            # Initialize pose extractor
            pose_extractor = PoseExtractor()
            
            # Create poses directory
            poses_dir = "/tmp/poses"
            
            # Extract poses from frames
            pose_files = pose_extractor.extract_poses_from_frame_list(frame_files, poses_dir)
            poses_extracted = len(pose_files)
            
            logger.info(f"Extracted {poses_extracted} poses")
            
            # Store pose files info for Lambda handler
            if pose_files:
                os.environ['POSE_FILES'] = json.dumps(pose_files)
        # ============= END POSE EXTRACTION =============
        # Annotate video
        logger.info("Adding annotations to video...")
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (width-10, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            # Add emotion text
            text = f"Emotion: {dominant_emotion.upper()}"
            if annotations:
                ann_text = ', '.join([f"{a['person']}:{a['emotion']}" for a in annotations])
                text += f" | {ann_text}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add transcription
            if transcription:
                # Show current segment based on time
                current_time = frame_count / fps
                current_text = ""
                for segment in segments:
                    if segment['start'] <= current_time <= segment['end']:
                        current_text = segment['text']
                        break
                
                if current_text:
                    # Wrap text if too long
                    words = current_text.split()
                    lines = []
                    current_line = []
                    for word in words:
                        current_line.append(word)
                        if len(' '.join(current_line)) > 60:
                            lines.append(' '.join(current_line[:-1]))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    y_offset = 70
                    for line in lines[:2]:  # Show max 2 lines
                        cv2.putText(frame, line, (20, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                        y_offset += 25
            
            # Add progress bar
            progress = int((frame_count / (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)) * (width - 40))
            cv2.rectangle(frame, (20, 110), (20 + progress, 115), (0, 255, 0), -1)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Create metadata
        video_name = os.path.basename(input_path)
        metadata = {
            'video_path': video_name,
            'transcription': transcription,
            'emotion': dominant_emotion,
            'duration_seconds': round(duration, 2),
            'emotion_keywords_found': ', '.join(emotions_found) if emotions_found else 'none',
            'annotations_found': ', '.join([f"{a['person']}:{a['emotion']}" for a in annotations]) if annotations else 'none',
            'frames_extracted': frames_extracted,
            'poses_extracted': poses_extracted,  # NEW
            'whisper_model': 'base',
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = output_path.replace('.mp4', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save CSV
        csv_path = output_path.replace('.mp4', '_analysis.csv')
        with open(csv_path, 'w') as f:
            f.write("video_path,transcription,emotion,duration_seconds,emotion_keywords_found,annotations_found,frames_extracted\n")
            f.write(f'"{video_name}","{transcription}","{dominant_emotion}",{duration:.2},"{metadata["emotion_keywords_found"]}","{metadata["annotations_found"]}",{frames_extracted}\n')
        
        # Store frame files info for Lambda handler
        if frame_files:
            os.environ['FRAME_FILES'] = json.dumps(frame_files)
        
        logger.info(f"Processing complete: {dominant_emotion} emotion, {frames_extracted} frames")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback: just copy the file
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path