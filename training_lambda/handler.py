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

def get_local_file_content(filename):
    """Read file from Lambda package"""
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return get_placeholder_content(filename)
    
def get_actual_file_from_s3(filename):
    """Get actual training file from S3"""
    try:
        response = s3.get_object(
            Bucket='patients999',
            Key=f'source-code/{filename}'
        )
        return response['Body'].read()
    except Exception as e:
        print(f"Error reading {filename} from S3: {e}")
        # Fallback to placeholder if file not found
        return get_placeholder_content(filename)

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
            'train_sagemaker.py': get_actual_file_from_s3('train_sagemaker.py'),
            'train.py': get_actual_file_from_s3('train.py'),
            'extract.py': get_actual_file_from_s3('extract.py'),  # Should be your real extract.py
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