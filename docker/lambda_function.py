import os
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import json
import boto3
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


def get_next_pose_index(bucket, prefix):
    """
    Find the highest existing pose index in the given S3 prefix and return the next available index.
    This prevents overwriting existing pose files when processing new videos.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix path (e.g., "patient_name/emotion/")
    
    Returns:
        int: The next available index to use for pose files
    """
    try:
        # List all objects in the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket,
            Prefix=prefix
        )
        
        max_index = -1
        
        # Iterate through all objects to find the highest index
        for page in page_iterator:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]
                
                # Check if it's a pose file with our naming convention
                if filename.startswith('pose_') and filename.endswith('.json'):
                    try:
                        # Extract the index from pose_XXXXXX.json
                        index_str = filename[5:11]  # Get the 6 digits
                        index = int(index_str)
                        max_index = max(max_index, index)
                    except (ValueError, IndexError):
                        # Skip files that don't match our expected format
                        continue
        
        # Return the next available index
        next_index = max_index + 1
        logger.info(f"Found {max_index + 1} existing pose files in {prefix}, starting at index {next_index}")
        return next_index
        
    except Exception as e:
        logger.warning(f"Error checking existing pose files in {prefix}: {e}")
        # If there's an error, start from 0 (but log the warning)
        return 0


def lambda_handler(event, context):
    """
    Lambda function triggered by S3 event when a video is uploaded.
    Downloads the video, processes it with emotion analysis and pose extraction, 
    and uploads results back to S3 with cumulative indexing to prevent overwrites.
    
    Expected S3 structure:
    Input: patient-name/video-file.mp4
    Output: annotated/patient-name/video-file.mp4
    Training: patients999/patient-name/emotion/pose_*.json
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
            key_lower = key.lower()
            if not any(key_lower.endswith(ext) for ext in video_extensions):
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
            
            # Get file extension for proper handling
            _, file_ext = os.path.splitext(key)
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_input:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_output:
                    # Initialize file paths
                    metadata_path = None
                    csv_path = None
                    
                    try:
                        # Download video from S3
                        logger.info("Downloading video from S3...")
                        s3_client.download_file(bucket, key, tmp_input.name)
                        
                        # Process video with emotion analysis
                        logger.info("Processing video with emotion analysis and pose extraction...")
                        result = process_video(tmp_input.name, tmp_output.name)
                        logger.info(f"process_video returned type: {type(result)}")
                        logger.info(f"result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")

                        annotated_video_path = result['output_path']
                        logger.info(f"annotated_video_path type: {type(annotated_video_path)}, value: {annotated_video_path}")
                        
                        # Get metadata for emotion detection
                        metadata = result.get('metadata', {})
                        
                        # Create output key maintaining patient folder structure
                        output_key = f"annotated/{key}"
                        
                        # Determine content type based on extension
                        content_type_map = {
                            '.mp4': 'video/mp4',
                            '.mov': 'video/quicktime',
                            '.avi': 'video/x-msvideo',
                            '.mkv': 'video/x-matroska',
                            '.webm': 'video/webm',
                            '.flv': 'video/x-flv'
                        }
                        video_content_type = content_type_map.get(file_ext.lower(), 'video/mp4')
                        
                        # Upload processed video back to S3
                        logger.info(f"Uploading annotated video to: {output_key}")
                        s3_client.upload_file(
                            annotated_video_path,
                            bucket,
                            output_key,
                            ExtraArgs={'ContentType': video_content_type}
                        )
                        
                        # Fix: Use proper file extension handling for metadata and CSV
                        base_output_path = os.path.splitext(annotated_video_path)[0]
                        base_output_key = os.path.splitext(output_key)[0]
                        
                        # Upload metadata if it exists
                        metadata_path = f"{base_output_path}_metadata.json"
                        if os.path.exists(metadata_path):
                            metadata_key = f"{base_output_key}_metadata.json"
                            s3_client.upload_file(
                                metadata_path,
                                bucket,
                                metadata_key,
                                ExtraArgs={'ContentType': 'application/json'}
                            )
                            logger.info(f"Uploaded metadata to: {metadata_key}")
                        
                        # Upload CSV analysis if it exists
                        csv_path = f"{base_output_path}_analysis.csv"
                        if os.path.exists(csv_path):
                            csv_key = f"{base_output_key}_analysis.csv"
                            s3_client.upload_file(
                                csv_path,
                                bucket,
                                csv_key,
                                ExtraArgs={'ContentType': 'text/csv'}
                            )
                            logger.info(f"Uploaded CSV analysis to: {csv_key}")
                        
                        # Upload extracted frames if they exist
                        if 'FRAME_FILES' in os.environ and os.environ['FRAME_FILES'].strip():
                            try:
                                frame_files = json.loads(os.environ['FRAME_FILES'])
                                video_basename = os.path.splitext(os.path.basename(key))[0]
                                
                                for frame_path, frame_filename in frame_files:
                                    if os.path.exists(frame_path):
                                        try:
                                            # Create frame key
                                            frame_key = f"annotated/{patient_name}/frames/{video_basename}/{frame_filename}"
                                            s3_client.upload_file(
                                                frame_path,
                                                bucket,
                                                frame_key,
                                                ExtraArgs={'ContentType': 'image/jpeg'}
                                            )
                                            logger.info(f"Uploaded frame to: {frame_key}")
                                        except Exception as e:
                                            logger.error(f"Error uploading frame {frame_filename}: {e}")
                                
                                # Clean up environment variable
                                del os.environ['FRAME_FILES']
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing FRAME_FILES JSON: {e}")
                                logger.error(f"FRAME_FILES content: {repr(os.environ.get('FRAME_FILES', 'NOT_SET'))}")
                            except Exception as e:
                                logger.error(f"Error processing frame files: {e}")
                        
                        # Upload extracted poses with CUMULATIVE INDEXING
                        if 'POSE_FILES' in os.environ and os.environ['POSE_FILES'].strip():
                            try:
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
                                if metadata_path and os.path.exists(metadata_path):
                                    try:
                                        with open(metadata_path, 'r') as f:
                                            metadata = json.load(f)
                                            dominant_emotion = metadata.get('emotion', 'neutral')
                                    except Exception as e:
                                        logger.warning(f"Could not read metadata for emotion: {e}")
                                
                                # Map to training bucket format
                                emotion_dir = emotion_mapping.get(dominant_emotion, dominant_emotion)
                                
                                # === CUMULATIVE INDEXING FIX ===
                                # Get the next available index for this patient/emotion combination
                                training_prefix = f"{patient_name}/{emotion_dir}/"
                                start_index = get_next_pose_index('patients999', training_prefix)
                                logger.info(f"Starting pose upload at index {start_index} for {training_prefix}")
                                
                                # Upload poses to BOTH locations with proper indexing
                                pose_upload_count = 0
                                for i, (pose_path, pose_filename) in enumerate(pose_files):
                                    if os.path.exists(pose_path):
                                        try:
                                            # Upload to annotated bucket for reference (keep original naming)
                                            annotated_pose_key = f"annotated/{patient_name}/poses/{video_basename}/{pose_filename}"
                                            s3_client.upload_file(
                                                pose_path,
                                                bucket,
                                                annotated_pose_key,
                                                ExtraArgs={'ContentType': 'application/json'}
                                            )
                                            logger.info(f"Uploaded pose to annotated: {annotated_pose_key}")
                                            
                                            # Upload to training bucket with CUMULATIVE index
                                            cumulative_index = start_index + i
                                            training_pose_key = f"{patient_name}/{emotion_dir}/pose_{cumulative_index:06d}.json"
                                            s3_client.upload_file(
                                                pose_path,
                                                'patients999',  # Training bucket
                                                training_pose_key,
                                                ExtraArgs={'ContentType': 'application/json'}
                                            )
                                            logger.info(f"Uploaded pose to training with cumulative index: {training_pose_key}")
                                            pose_upload_count += 1
                                            
                                        except Exception as e:
                                            logger.error(f"Error uploading pose {pose_filename}: {e}")
                                    else:
                                        logger.warning(f"Pose file not found: {pose_path}")
                                
                                logger.info(f"Successfully uploaded {pose_upload_count} pose files (indices {start_index} to {start_index + pose_upload_count - 1})")
                                
                                # Add session metadata to track which poses came from which video
                                session_metadata = {
                                    'video_file': key,
                                    'processing_time': result.get('metadata', {}).get('processing_time', 'unknown'),
                                    'emotion': emotion_dir,
                                    'pose_indices': {
                                        'start': start_index,
                                        'end': start_index + pose_upload_count - 1,
                                        'count': pose_upload_count
                                    }
                                }
                                
                                session_key = f"{patient_name}/{emotion_dir}/.sessions/{video_basename}_session.json"
                                s3_client.put_object(
                                    Bucket='patients999',
                                    Key=session_key,
                                    Body=json.dumps(session_metadata, indent=2),
                                    ContentType='application/json'
                                )
                                logger.info(f"Saved session metadata to: {session_key}")
                                
                                # Clean up environment variable
                                del os.environ['POSE_FILES']
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing POSE_FILES JSON: {e}")
                                logger.error(f"POSE_FILES content: {repr(os.environ.get('POSE_FILES', 'NOT_SET'))}")
                            except Exception as e:
                                logger.error(f"Error processing pose files: {e}")
                        
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
                            shutil.rmtree("/tmp/frames", ignore_errors=True)
                            
                        if os.path.exists("/tmp/poses"):
                            shutil.rmtree("/tmp/poses", ignore_errors=True)
                        
                        # Clean up environment variables
                        for var in ['S3_BUCKET', 'S3_KEY', 'FRAME_FILES', 'POSE_FILES']:
                            if var in os.environ:
                                del os.environ[var]
        
        return {
            'statusCode': 200,
            'body': json.dumps('Video processing with emotion analysis and pose extraction completed successfully')
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }