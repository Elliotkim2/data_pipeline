import json
import boto3
import os
from urllib.parse import unquote_plus
import tempfile
import logging

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

# Import the processor
from audio_emotion_extractor_whisper import process_video

def lambda_handler(event, context):
    """
    Lambda function triggered by S3 event when a video is uploaded.
    Downloads the video, processes it with emotion analysis, and uploads the result back to S3.
    
    Expected S3 structure:
    Input: patient-name/video-file.mp4
    Output: annotated/patient-name/video-file.mp4
    """
    
    try:
        # Parse S3 event
        for record in event['Records']:
            # Get bucket and object details
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])
            
            # Skip if file is already in annotated folder
            if key.startswith('annotated/'):
                logger.info(f"Skipping already annotated file: {key}")
                continue
            
            # Check if file is a video (add more extensions as needed)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
            if not any(key.lower().endswith(ext) for ext in video_extensions):
                logger.info(f"Skipping non-video file: {key}")
                continue
            
            logger.info(f"Processing video: {key} from bucket: {bucket}")
            
            # Parse the key to extract patient folder and filename
            path_parts = key.split('/')
            if len(path_parts) < 2:
                logger.warning(f"File not in expected patient-name/file format: {key}")
                continue
            
            # Extract patient name (first part) and rest of the path
            patient_name = path_parts[0]
            file_path = '/'.join(path_parts[1:])
            
            logger.info(f"Patient: {patient_name}, File: {file_path}")
            
            # Pass S3 info to processor via environment variables
            os.environ['S3_BUCKET'] = bucket
            os.environ['S3_KEY'] = key
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1]) as tmp_input:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1]) as tmp_output:
                    # Initialize file paths
                    metadata_path = None
                    csv_path = None
                    
                    try:
                        # Download video from S3
                        logger.info("Downloading video from S3...")
                        s3_client.download_file(bucket, key, tmp_input.name)
                        
                        # Process video with emotion analysis
                        logger.info("Processing video with emotion analysis...")
                        annotated_video_path = process_video(tmp_input.name, tmp_output.name)
                        
                        # Create output key maintaining patient folder structure
                        # Input: patient-name/video.mp4 -> Output: annotated/patient-name/video.mp4
                        output_key = f"annotated/{key}"
                        
                        # Upload processed video back to S3
                        logger.info(f"Uploading annotated video to: {output_key}")
                        s3_client.upload_file(
                            annotated_video_path,
                            bucket,
                            output_key,
                            ExtraArgs={'ContentType': 'video/mp4'}
                        )
                        
                        # Upload metadata if it exists
                        metadata_path = annotated_video_path.replace('.mp4', '_metadata.json')
                        if os.path.exists(metadata_path):
                            metadata_key = output_key.replace('.mp4', '_metadata.json')
                            s3_client.upload_file(
                                metadata_path,
                                bucket,
                                metadata_key,
                                ExtraArgs={'ContentType': 'application/json'}
                            )
                            logger.info(f"Uploaded metadata to: {metadata_key}")
                        
                        # Upload CSV analysis if it exists
                        csv_path = annotated_video_path.replace('.mp4', '_analysis.csv')
                        if os.path.exists(csv_path):
                            csv_key = output_key.replace('.mp4', '_analysis.csv')
                            s3_client.upload_file(
                                csv_path,
                                bucket,
                                csv_key,
                                ExtraArgs={'ContentType': 'text/csv'}
                            )
                            logger.info(f"Uploaded CSV analysis to: {csv_key}")
                        
                        # Upload extracted frames if they exist
                        if 'FRAME_FILES' in os.environ:
                            frame_files = json.loads(os.environ['FRAME_FILES'])
                            video_basename = os.path.splitext(os.path.basename(key))[0]
                            
                            for frame_path, frame_filename in frame_files:
                                if os.path.exists(frame_path):
                                    # Create frame key: annotated/patient-name/frames/video-name/frame_001.jpg
                                    frame_key = f"annotated/{patient_name}/frames/{video_basename}/{frame_filename}"
                                    s3_client.upload_file(
                                        frame_path,
                                        bucket,
                                        frame_key,
                                        ExtraArgs={'ContentType': 'image/jpeg'}
                                    )
                                    logger.info(f"Uploaded frame to: {frame_key}")
                            
                            # Clean up environment variable
                            del os.environ['FRAME_FILES']
                        if 'POSE_FILES' in os.environ:
                            pose_files = json.loads(os.environ['POSE_FILES'])
                            video_basename = os.path.splitext(os.path.basename(key))[0]
                            
                            # Map emotion names to training format
                            emotion_mapping = {
                                'happy': 'happiness',
                                'sad': 'sadness', 
                                'angry': 'anger',
                                'surprised': 'surprise',
                                'neutral': 'neutral'
                            }
                            
                            # Get emotion from metadata if available
                            dominant_emotion = 'neutral'
                            if os.path.exists(metadata_path):
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                    dominant_emotion = metadata.get('emotion', 'neutral')
                            
                            # Map to training bucket format
                            emotion_dir = emotion_mapping.get(dominant_emotion, dominant_emotion)
                            
                            # Upload poses to BOTH locations:
                            # 1. Annotated bucket for reference
                            # 2. Training bucket for model training
                            
                            for i, (pose_path, pose_filename) in enumerate(pose_files):
                                if os.path.exists(pose_path):
                                    # Upload to annotated bucket for reference
                                    annotated_pose_key = f"annotated/{patient_name}/poses/{video_basename}/{pose_filename}"
                                    s3_client.upload_file(
                                        pose_path,
                                        bucket,
                                        annotated_pose_key,
                                        ExtraArgs={'ContentType': 'application/json'}
                                    )
                                    logger.info(f"Uploaded pose to annotated: {annotated_pose_key}")
                                    
                                    # Upload to training bucket in correct format
                                    # Use index-based naming for training compatibility
                                    training_pose_key = f"{patient_name}/{emotion_dir}/pose_{i:06d}.json"
                                    s3_client.upload_file(
                                        pose_path,
                                        'patients999',  # Training bucket
                                        training_pose_key,
                                        ExtraArgs={'ContentType': 'application/json'}
                                    )
                                    logger.info(f"Uploaded pose to training: {training_pose_key}")
                            
                            # Clean up environment variable
                            del os.environ['POSE_FILES']
                        
                        logger.info(f"Successfully processed and uploaded: {output_key}")
                        
                    finally:
                        # Clean up temporary files
                        
                        if os.path.exists(tmp_input.name):
                            os.unlink(tmp_input.name)
                        if os.path.exists(tmp_output.name):
                            os.unlink(tmp_output.name)
                        
                        # Clean up metadata and csv files if they exist
                        if metadata_path and os.path.exists(metadata_path):
                            os.unlink(metadata_path)
                        if csv_path and os.path.exists(csv_path):
                            os.unlink(csv_path)
                        
                        # Clean up frames directory
                        import shutil
                        if os.path.exists("/tmp/frames"):
                            shutil.rmtree("/tmp/frames")
                            
                        if os.path.exists("/tmp/poses"):
                            shutil.rmtree("/tmp/poses")
        return {
            'statusCode': 200,
            'body': json.dumps('Video processing with emotion analysis completed successfully')
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
