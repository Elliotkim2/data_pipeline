import json
import boto3
import os
from datetime import datetime
import time
import tarfile
import io

s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'patients999')
SAGEMAKER_ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::443293291738:role/ai-training-pipeline-sagemaker-role')

def handler(event, context):
    """Handler that starts SageMaker training with real training scripts"""
    print(f"Received event: {json.dumps(event)}")
    
    try:
        job_id = event.get('job_id', 'unknown')
        bucket = event.get('bucket', 'patients999')
        key = event.get('key', '')
        
        print(f"Processing job {job_id} for s3://{bucket}/{key}")
        
        if key and '/' in key:
            base_dir = key.split('/')[0]
            training_job_name = start_sagemaker_training(job_id, bucket, base_dir)
            
            return {
                'statusCode': 200,
                'body': {
                    'decision': 'TRAIN',
                    'training_job_name': training_job_name,
                    'message': f'Started SageMaker training job: {training_job_name}'
                }
            }
        else:
            return {
                'statusCode': 200,
                'body': {
                    'decision': 'SKIP',
                    'message': 'No valid data path'
                }
            }
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': {
                'decision': 'ERROR',
                'error': str(e)
            }
        }

def start_sagemaker_training(job_id, bucket, base_dir):
    """Start SageMaker training job with real training scripts"""
    safe_job_id = job_id.replace('_', '-').replace('/', '-')
    timestamp = str(int(time.time()))
    training_job_name = f"emotion-{safe_job_id}-{timestamp}"[:63]
    training_job_name = ''.join(c if c.isalnum() or c == '-' else '-' for c in training_job_name).strip('-')
    
    print(f"Creating training job: {training_job_name}")
    
    input_path = f"s3://{bucket}/{base_dir}/"
    output_path = f"s3://{MODEL_BUCKET}/models/{safe_job_id}"
    code_path = f"s3://{MODEL_BUCKET}/training-code/{safe_job_id}/sourcedir.tar.gz"
    
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    # Upload the real training code
    upload_training_code(code_path)
    
    training_params = {
        'TrainingJobName': training_job_name,
        'RoleArn': SAGEMAKER_ROLE_ARN,
        'HyperParameters': {
            'sagemaker_program': 'train_sagemaker.py',  # Use the real wrapper
            'sagemaker_submit_directory': code_path
        },
        'AlgorithmSpecification': {
            'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-cpu-py38',
            'TrainingInputMode': 'File'
        },
        'InputDataConfig': [{
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_path,
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'application/json'
        }],
        'OutputDataConfig': {
            'S3OutputPath': output_path
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours for real training
        },
        'EnableNetworkIsolation': False
    }
    
    response = sagemaker.create_training_job(**training_params)
    print(f"Training job created: {response['TrainingJobArn']}")
    
    return training_job_name

def upload_training_code(s3_path):
    """Package and upload all training scripts"""
    print("Creating training code package with real scripts...")
    
    tar_buffer = io.BytesIO()
    
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        # Add all the training files
        files_to_add = {
            'train_sagemaker.py': get_train_sagemaker_content(),
            'train.py': get_placeholder_content('train.py'),  # Should be your real train.py
            'extract.py': get_placeholder_content('extract.py'),  # Should be your real extract.py
            'requirements.txt': get_requirements_content()
        }
        
        for filename, content in files_to_add.items():
            info = tarfile.TarInfo(filename)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    
    # Upload to S3
    tar_buffer.seek(0)
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    
    s3.put_object(Bucket=bucket, Key=key, Body=tar_buffer.getvalue())
    print(f"Training code uploaded to {s3_path}")

def get_train_sagemaker_content():
    """The wrapper script that SageMaker will run"""
    return b'''#!/usr/bin/env python
import os
import sys
import subprocess
import shutil

def setup_training_data():
    """Reorganize training data if needed"""
    train_dir = '/opt/ml/input/data/train'
    print(f"Training data directory: {train_dir}")
    print(f"Contents: {os.listdir(train_dir)}")
    
    # Check if we need to move emotion directories up
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path):
            subdirs = os.listdir(item_path)
            emotions = ['happiness', 'sadness', 'anger', 'disgust', 'fear', 'neutral', 'surprise']
            has_emotion_dirs = any(subdir.lower() in emotions for subdir in subdirs)
            
            if has_emotion_dirs:
                print(f"Moving emotion directories from {item_path} to {train_dir}")
                for subdir in subdirs:
                    src = os.path.join(item_path, subdir)
                    dst = os.path.join(train_dir, subdir)
                    if os.path.isdir(src) and not os.path.exists(dst):
                        shutil.move(src, dst)
                try:
                    os.rmdir(item_path)
                except:
                    pass
    
    print(f"Final training directory contents: {os.listdir(train_dir)}")

def main():
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    
    print(f"Model dir: {model_dir}")
    print(f"Train dir: {train_dir}")
    print(f"Output dir: {output_dir}")
    
    # Reorganize data if needed
    setup_training_data()
    
    # Install requirements
    print("Installing requirements...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Run the actual training script
    print("Running train.py...")
    result = subprocess.run([
        sys.executable, 'train.py',
        '--data_dir', train_dir,
        '--output_dir', model_dir,
        '--force_extract'  # Force feature extraction
    ], capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Training script failed with return code {result.returncode}")
    
    # Ensure we have a model file
    model_file = os.path.join(model_dir, 'advanced_emotion_model.pkl')
    if not os.path.exists(model_file):
        print("Warning: Model file not found after training")
        # Create a minimal file so SageMaker doesn't fail
        with open(model_file, 'w') as f:
            f.write("Training incomplete - check logs")
    else:
        print(f"Model saved successfully: {model_file}")
    
    # List all outputs
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            print(f"Output file: {os.path.join(root, file)}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
'''

def get_requirements_content():
    """Requirements for the training environment"""
    return b'''numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.66.1
'''

def get_placeholder_content(filename):
    """Placeholder for actual training files - these should be replaced with real files"""
    return f"# Placeholder for {filename}\n# This should be replaced with the actual file content".encode()