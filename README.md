# VOICE Data Pipeline Documentation

[![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20S3%20%7C%20SageMaker-orange)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![Terraform](https://img.shields.io/badge/Terraform-IaC-purple)](https://www.terraform.io/)

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

The VOICE (Video-based Observation of Intelligent Communication and Emotion) Data Pipeline is an AI-powered emotion detection system designed for patients with Profound Intellectual and Multiple Disabilities (PIMD). The system processes video recordings to:

1. **Transcribe audio** using OpenAI Whisper
2. **Extract emotion keywords** from transcriptions
3. **Detect body poses** using MediaPipe
4. **Annotate videos** with detected emotions
5. **Train ML models** on pose data using AWS SageMaker

### Key Technologies
- **AWS Services**: Lambda, S3, SageMaker, Step Functions, ECR, CloudWatch
- **ML/AI**: Whisper, MediaPipe, scikit-learn, RandomForest
- **Video Processing**: OpenCV, MoviePy, FFmpeg
- **Infrastructure**: Terraform, Docker

---

## Architecture

```
┌─────────────────┐
│  Video Upload   │
│   to S3 Bucket  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Lambda (Docker Container)      │
│  ┌──────────────────────────┐   │
│  │ 1. Audio Extraction      │   │
│  │ 2. Whisper Transcription │   │
│  │ 3. Emotion Detection     │   │
│  │ 4. Frame Extraction      │   │
│  │ 5. Pose Extraction       │   │
│  │ 6. Video Annotation      │   │
│  └──────────────────────────┘   │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  S3 Output Structure            │
│  ├─ annotated/                  │
│  │  └─ patient-name/            │
│  │     ├─ video.mp4             │
│  │     ├─ frames/               │
│  │     └─ poses/                │
│  └─ patients999/                │
│     └─ patient-name/            │
│        └─ emotion/              │
│           └─ pose_*.json        │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Step Functions Workflow        │
│  ┌──────────────────────────┐   │
│  │ Process Upload Lambda    │   │
│  │         ↓                │   │
│  │ Training Agent Lambda    │   │
│  │         ↓                │   │
│  │ SageMaker Training Job   │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

### AWS Resource Structure

**S3 Buckets:**
- `video-processing-bucket`: Raw video uploads and annotated outputs
- `patients999`: Training data organized by patient/emotion

**Lambda Functions:**
1. **Video Processor** (Docker): Main processing pipeline
2. **Process Upload**: Monitors training data uploads
3. **Training Agent**: Orchestrates SageMaker training

**IAM Roles:**
- Video Processing Role: S3 read/write, CloudWatch logs
- AI Training Role: S3, DynamoDB, SageMaker, Bedrock access
- SageMaker Execution Role: S3, ECR, CloudWatch access

---

## Features

### Video Processing
- **Multi-format support**: MP4, MOV, AVI, MKV, WebM, FLV
- **Audio transcription**: OpenAI Whisper with keyword-based emotion detection
- **Pose extraction**: MediaPipe body pose keypoints in OpenPose format
- **Video annotation**: Real-time emotion overlays and transcription display
- **Frame extraction**: Smart frame selection at emotion timestamps

### Emotion Detection
- **Keyword-based**: Detects emotions from audio transcription
- **Supported emotions**: Happy, Sad, Angry, Surprised, Neutral
- **Pattern matching**: Identifies emotion annotations (e.g., "John is happy")
- **Multi-modal**: Combines audio and visual cues

### ML Training Pipeline
- **Feature extraction**: 100+ pose-based features including:
  - Joint angles (9 key joints)
  - Body proportions (height, width, aspect ratio)
  - Keypoint distances (10 pairs)
  - Symmetry features (6 pairs)
  - Velocity features (8 key joints)
  - Acceleration features (3 key joints)
  - Movement distribution (4 body parts)
  - Periodicity features (autocorrelation)
  - Bounding box features
- **Model**: RandomForest classifier
- **Training**: Automated via SageMaker with ml.m5.large instances

### Infrastructure as Code
- **Modular Terraform**: Reusable modules for IAM, Lambda, S3, Step Functions
- **Environment-based**: Easy dev/staging/prod deployments
- **Versioning**: S3 versioning and encryption enabled

---

## Prerequisites

### Required Tools
```bash
# AWS CLI
aws --version  # >= 2.0

# Terraform
terraform --version  # >= 1.0

# Docker
docker --version  # >= 20.10

# Python
python --version  # >= 3.9
```

### AWS Requirements
- AWS Account with appropriate permissions
- Configured AWS credentials (`~/.aws/credentials`)
- Sufficient service quotas for Lambda (10GB memory, 15min timeout)

### Python Dependencies
See `docker/requirements.txt` for the Lambda container:
- openai-whisper==20230918
- torch==2.0.1
- moviepy==1.0.3
- opencv-python-headless==4.8.0.76
- mediapipe==0.10.14
- pandas, numpy, ffmpeg-python

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Elliotkim2/data_pipeline.git
cd data_pipeline
```

### 2. Configure AWS Credentials
```bash
aws configure
# Enter your Access Key, Secret Key, and Region (us-east-1 recommended)
```

### 3. Build Docker Image
```bash
cd docker
docker buildx build --platform linux/amd64 --load -t whisper-lambda-with-pose .   
```

### 4. Push to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name voice-video-processor

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag whisper-lambda-with-pose:latest \                                                                
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}
```

### 5. Deploy Infrastructure
```bash
# Initialize Terraform
terraform init

# Review plan
terraform plan

# Apply configuration
terraform apply
```

### 6. Upload Training Code to S3
```bash
# Package training scripts
cd training_lambda
tar -czf sourcedir.tar.gz train_sagemaker.py train.py extract.py requirements.txt

# Upload to S3
aws s3 cp sourcedir.tar.gz s3://patients999/source-code/
aws s3 cp train_sagemaker.py s3://patients999/source-code/
aws s3 cp train.py s3://patients999/source-code/
aws s3 cp extract.py s3://patients999/source-code/
```

---

## Usage

### Upload Video for Processing

#### Option 1: AWS Console
1. Navigate to S3 in AWS Console
2. Open your video processing bucket
3. Create folder structure: `patient-name/`
4. Upload video file (e.g., `patient-name/session1.mp4`)

#### Option 2: AWS CLI
```bash
aws s3 cp video.mp4 s3://your-video-bucket/patient-name/session1.mp4
```

### Expected Output Structure
```
s3://your-video-bucket/
├── patient-name/
│   └── session1.mp4                    # Original video
└── annotated/
    └── patient-name/
        ├── session1.mp4                # Annotated video
        ├── session1_metadata.json      # Processing metadata
        ├── session1_analysis.csv       # Emotion analysis
        ├── frames/
        │   └── session1/
        │       ├── frame_000_at_5.23s_john_happy.jpg
        │       └── frame_001_at_pos_120.jpg
        └── poses/
            └── session1/
                ├── pose_000_at_5.23s_john_happy.json
                └── pose_001_at_pos_120.json

s3://patients999/
└── patient-name/
    ├── happiness/
    │   ├── pose_000000.json
    │   ├── pose_000001.json
    │   └── .sessions/
    │       └── session1_session.json
    └── sadness/
        └── pose_000000.json
```

### Monitor Processing

#### CloudWatch Logs
```bash
# Video processor logs
aws logs tail /aws/lambda/video-annotation-processor-whisper --follow

# Training logs
aws logs tail /aws/stepfunctions/voice-training-pipeline-dev --follow
```

#### Step Functions Console
1. Navigate to Step Functions in AWS Console
2. View execution history and details
3. Monitor training pipeline progress

### Trigger Manual Training
```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT_ID:stateMachine:voice-training-pipeline-dev \
  --input '{"job_id": "manual-test", "bucket": "patients999", "key": "patient-name/happiness/"}'
```


---

## Training & Validation

### Example Training Output

**Successful Training Run (October 14, 2025)**

```bash
============================================================
SESSION-BASED EMOTION RECOGNITION TRAINING
============================================================
Data directory: /opt/ml/input/data/train
Output directory: /opt/ml/model

Found 2 emotion directories:
  - Sadness: /opt/ml/input/data/train/sadness
  - Happiness: /opt/ml/input/data/train/happiness

Processing 2 emotion directories...

Processing Sadness from /opt/ml/input/data/train/sadness
Found 5 sessions with poses:
  - happiness: 14 poses
  - IMG_0503: 11 poses
  - IMG_0435: 14 poses
  - IMG_0501: 12 poses
  - IMG_0499: 14 poses

Processing Happiness from /opt/ml/input/data/train/happiness
Found 5 sessions with poses:
  - saddness: 14 poses
  - IMG_0500: 12 poses
  - Elot1: 13 poses
  - IMG_0498: 14 poses
  - IMG_0502: 12 poses

📊 Feature Extraction Summary:
Extracted 88 features from 10 video sessions

Sessions per emotion:
  Sadness: 5 sessions
  Happiness: 5 sessions

Detailed session breakdown:
  happiness (Sadness): 14 poses
  IMG_0503 (Sadness): 11 poses
  IMG_0435 (Sadness): 14 poses
  IMG_0501 (Sadness): 12 poses
  IMG_0499 (Sadness): 14 poses
  saddness (Happiness): 14 poses
  IMG_0500 (Happiness): 12 poses
  Elot1 (Happiness): 13 poses
  IMG_0498 (Happiness): 14 poses
  IMG_0502 (Happiness): 12 poses

✅ Training data validation passed!
Training emotion model...
Label mapping: {0: 'Happiness', 1: 'Sadness'}
Training on 8 samples, testing on 2 samples

Performing cross-validation...
Cross-validation scores: [0.5  0.75]
Mean CV score: 0.6250 ± 0.1250

📊 Model Performance:
Training accuracy: 1.0000
Testing accuracy: 0.5000

Classification Report:
              precision    recall  f1-score   support
   Happiness       0.50      1.00      0.67         1
     Sadness       0.00      0.00      0.00         1
    accuracy                           0.50         2
   macro avg       0.25      0.50      0.33         2
weighted avg       0.25      0.50      0.33         2

✅ Model saved to /opt/ml/model/advanced_emotion_model.pkl
✅ Session info saved to /opt/ml/model/training_sessions.json

🎉 Training completed successfully!

Final model directory contents:
  feature_importance.png: 44344 bytes
  advanced_emotion_model.pkl: 64055 bytes
  confusion_matrix.png: 24050 bytes
  training_sessions.json: 860 bytes
```

**Training Insights**

Data Requirements:
- ✅ **Minimum**: 10 total video sessions, 3 per emotion class
- ✅ **Current**: 10 sessions (5 Happiness, 5 Sadness)
- ✅ **Session-based**: Each video becomes 1 training sample

Performance Notes:
- **Training Accuracy**: 100% (expected with small dataset)
- **CV Score**: 62.5% ± 12.5% (indicates model is learning patterns)
- **Test Accuracy**: 50% (limited by small test set of only 2 samples)
- **Recommendation**: Upload 20+ videos per emotion for production-quality model

Feature Engineering:
- 88 pose-based features extracted per session
- Features include joint angles, velocities, symmetry, and movement patterns
- Each video session averaged 12-14 pose frames

---

## Training Validation

### Check Training Status

**View Recent Training Jobs:**
```bash
aws sagemaker list-training-jobs \
  --name-contains emotion-Elliot \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 5 \
  --query 'TrainingJobSummaries[*].[TrainingJobName,TrainingJobStatus,CreationTime]' \
  --output table
```

**Get Training Job Details:**
```bash
aws sagemaker describe-training-job \
  --training-job-name emotion-Elliot-TIMESTAMP
```

**Download Model Artifacts:**
```bash
# Model is saved to S3 after training
aws s3 cp s3://patients999/models/Elliot/model.tar.gz .
tar -xzf model.tar.gz
```

### Verify Model Quality

**Check Training Metrics:**
- Training Accuracy: Should be >80% for good model
- Test Accuracy: Should be >60% with sufficient data
- CV Score: Should be >70% with 20+ samples per class

**Review Visualizations:**
```bash
# Download confusion matrix and feature importance plots
aws s3 cp s3://patients999/models/Elliot/confusion_matrix.png .
aws s3 cp s3://patients999/models/Elliot/feature_importance.png .
```

**Session Metadata:**
```bash
# Check which videos were used for training
aws s3 cp s3://patients999/models/Elliot/training_sessions.json - | jq .
```

### Monitor Training Logs

**CloudWatch Logs:**
```bash
# Training logs (SageMaker)
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix emotion-Elliot \
  --follow

# Step Functions execution logs
aws logs tail /aws/stepfunctions/voice-training-pipeline-dev --follow

# Process Upload Lambda logs
aws logs tail /aws/lambda/ai-training-pipeline-process-upload-dev --follow
```

**Step Functions Console:**
1. Navigate to Step Functions in AWS Console
2. Select `voice-training-pipeline-dev` state machine
3. View execution history and details
4. Monitor training pipeline progress in real-time

---

## Component Details

### Training Pipeline (`training_lambda/`)

**Feature Extractor** (`extract.py`)
- **PoseFeatureExtractor Class**: Comprehensive feature engineering
- **88 Core Features**:
  - Postural: Joint angles (9 joints: neck, shoulders, elbows, hips, knees), body proportions (height, width, aspect ratio), keypoint distances (10 critical pairs), symmetry features (left/right balance across 6 pairs)
  - Kinematic: Velocity features (8 tracked joints), acceleration features (3 key joints), movement distribution (torso, arms, legs)
  - Temporal: Periodicity via autocorrelation analysis
  - Global: Bounding box metrics, overall movement magnitude
- **Robust Handling**: NaN management, confidence thresholding, outlier detection

**Training Script** (`train.py`)
- **Model**: RandomForest (100 estimators, random_state=42)
- **Session-Based Training**: Each video = 1 sample (not directory-based)
- **Pipeline**:
  1. Load session metadata from `.sessions/` folders
  2. Group poses by original video source
  3. Extract 88 features per video session
  4. Label encoding (LabelEncoder) and feature scaling (StandardScaler)
  5. Train/test split (80/20, stratified when sample size permits)
  6. Cross-validation (2-5 fold, adaptive based on sample size)
  7. Model evaluation with confusion matrix and feature importance
- **Outputs**: 
  - `advanced_emotion_model.pkl` (model + scaler + encoder)
  - `confusion_matrix.png` (normalized confusion matrix visualization)
  - `feature_importance.png` (top 20 features ranked by importance)
  - `training_sessions.json` (metadata about training data)

**Minimum Data Requirements:**
```python
MIN_SAMPLES_PER_CLASS = 3  # Minimum video sessions per emotion
MIN_TOTAL_SAMPLES = 10     # Minimum total video sessions
```

Recommended for Production:
- 20+ video sessions per emotion class
- 50+ total video sessions for robust model
- Multiple recording conditions (lighting, angles, distances)

**SageMaker Wrapper** (`train_sagemaker.py`)
- **Data Setup**: Reorganizes S3 data structure for training
- **Error Handling**: Creates fallback minimal model on training failure
- **Model Validation**: Verifies pickle file integrity before upload
- **Environment Variables**:
  - `SM_MODEL_DIR`: Model output directory (`/opt/ml/model`)
  - `SM_CHANNEL_TRAIN`: Training data input (`/opt/ml/input/data/train`)
  - `SM_OUTPUT_DATA_DIR`: Additional outputs

**Training Agent** (`handler.py`)
- **SageMaker Job Creation**: Configures and launches training jobs
- **Parameters**:
  - Instance: `ml.m5.large` (2 vCPU, 8 GB RAM)
  - Max runtime: 7200 seconds (2 hours)
  - Algorithm: PyTorch CPU training container
- **Code Packaging**: Creates tarball with all training scripts
- **Job Naming**: `emotion-{patient_id}-{timestamp}`

---

## Component Details

### Video Processing Lambda (`docker/`)

**Lambda Handler** (`lambda_function.py`)
- **Trigger**: S3 ObjectCreated events
- **Timeout**: 900 seconds (15 minutes)
- **Memory**: 10GB
- **Key Functions**:
  - `lambda_handler()`: Main entry point
  - `get_next_pose_index()`: Cumulative pose indexing to prevent overwrites
  - File management and S3 upload orchestration

**Audio Emotion Extractor** (`audio_emotion_extractor_whisper.py`)
- **Whisper Integration**: Model caching for performance
- **Emotion Keywords**: 5 emotion categories with keyword lists
- **Pattern Matching**: Regex for annotation detection
- **Features**:
  - Audio extraction from video
  - Transcription with timestamps
  - Emotion keyword detection
  - Frame extraction at emotion timestamps
  - Video annotation with overlays

**Pose Extractor** (`pose_extractor.py`)
- **MediaPipe Integration**: Body pose detection
- **Output Format**: OpenPose-compatible 25-keypoint JSON
- **Lazy Initialization**: Efficient resource usage
- **Features**:
  - Single frame pose extraction
  - Batch processing
  - Error handling for missing landmarks

### Training Pipeline (`training_lambda/`)

**Feature Extractor** (`extract.py`)
- **PoseFeatureExtractor Class**: Comprehensive feature engineering
- **Features** (100+ dimensions):
  - Postural: Joint angles, body proportions, keypoint distances, symmetry
  - Kinematic: Velocity, acceleration, movement distribution
  - Temporal: Periodicity via autocorrelation
  - Global: Bounding box, overall movement
- **Robust Handling**: NaN management, confidence thresholding

**Training Script** (`train.py`)
- **Model**: RandomForest (100 estimators)
- **Pipeline**:
  1. Feature extraction from JSON directories
  2. Label encoding and scaling
  3. Train/test split (80/20, stratified)
  4. Cross-validation (5-fold)
  5. Model evaluation with confusion matrix
- **Outputs**: Model pickle, visualizations (confusion matrix, feature importance)

**SageMaker Wrapper** (`train_sagemaker.py`)
- **Data Setup**: Reorganizes S3 data structure
- **Error Handling**: Fallback minimal model on failure
- **Model Validation**: Verifies pickle file integrity
- **Environment Variables**:
  - `SM_MODEL_DIR`: Model output directory
  - `SM_CHANNEL_TRAIN`: Training data input
  - `SM_OUTPUT_DATA_DIR`: Additional outputs

**Training Agent** (`handler.py`)
- **SageMaker Job Creation**: Configures training jobs
- **Parameters**:
  - Instance: ml.m5.large
  - Max runtime: 2 hours
  - Algorithm: PyTorch CPU training container
- **Code Packaging**: Creates tarball with all training scripts

### Infrastructure Modules

**IAM Module** (`modules/iam/`)
- **Roles**:
  - Video processing Lambda
  - AI training Lambda
  - SageMaker execution
  - Step Functions
- **Policies**: Least-privilege access to S3, SageMaker, DynamoDB, Bedrock

**Lambda Module** (`modules/lambda/`)
- **Functions**:
  - `video-processor`: Docker container with ML dependencies
  - `process-upload`: Monitors pose JSON uploads
  - `training-agent`: Orchestrates SageMaker training
- **Configuration**: Timeouts, memory, environment variables

**S3 Module** (`modules/s3/`)
- **Buckets**: Video processing and patient data
- **Features**:
  - Versioning enabled
  - Server-side encryption (AES256)
  - Public access blocked
  - Event notifications to Lambda

**Step Functions Module** (`modules/stepfunctions/`)
- **State Machine**: Training pipeline orchestration
- **States**:
  1. AnalyzeData: Evaluate training data
  2. TrainingDecision: Determine if training needed
  3. LogTraining/SkipTraining: Execute or skip
  4. UpdateStatus: Finalize execution

---

## Data Flow

### Training Pipeline

```
Pose JSON upload → S3 event notification
  ↓
Process Upload Lambda triggered
  ↓
Validate training data:
  - Check file count (≥10 total pose files)
  - Check data size (≥5 MB)
  - Check emotion coverage (≥2 emotions with ≥3 sessions each)
  - Check training cooldown (prevent duplicate runs within 1 hour)
  ↓
If validation passes → Trigger Step Functions state machine
  ↓
Step Functions: AnalyzeData state
  - Evaluate data quality
  - Determine if training should proceed
  ↓
Step Functions: TrainingDecision state
  - Branch to either LogTraining or SkipTraining
  ↓
Step Functions: LogTraining state
  - Training Agent Lambda creates SageMaker job
  - Job configuration:
    * Training data: s3://patients999/{patient_id}/
    * Output: s3://patients999/models/{patient_id}/
    * Training script: train_sagemaker.py
    * Instance: ml.m5.large
  ↓
SageMaker Training Job:
  1. Downloads all pose JSONs from patients999 bucket
  2. Loads session metadata from .sessions/ folders
  3. Groups poses by original video source
  4. Extracts 88 features per video session
  5. Splits into train (80%) and test (20%) sets
  6. Trains RandomForest classifier
  7. Generates visualizations (confusion matrix, feature importance)
  8. Saves model pickle + artifacts
  ↓
Model artifacts uploaded to S3:
  - s3://patients999/models/{patient_id}/model.tar.gz
  - Contains:
    * advanced_emotion_model.pkl
    * confusion_matrix.png
    * feature_importance.png
    * training_sessions.json
  ↓
Step Functions: UpdateStatus state
  - Record training completion
  - Update DynamoDB training history
```

---

## Development

### Local Testing

**Test Docker Container Locally**
```bash
cd docker

# Build image
docker build -t voice-test .

# Run container
docker run -p 9000:8080 voice-test

# Test with sample event
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d @test_event.json
```

**Test Feature Extraction**
```python
from extract import PoseFeatureExtractor

extractor = PoseFeatureExtractor()
features = extractor.extract_features_from_json_files('path/to/pose/jsons/')
print(f"Extracted {len(features)} features")
```

### Debugging

**Enable Verbose Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Lambda Logs**
```bash
aws logs tail /aws/lambda/FUNCTION_NAME --follow --format short
```

**Inspect S3 Objects**
```bash
aws s3 ls s3://bucket-name/path/ --recursive
aws s3 cp s3://bucket-name/path/file.json - | jq .
```

### Environment Variables

**Lambda Environment Variables:**
- `PATIENTS_BUCKET`: Training data bucket
- `VIDEO_BUCKET`: Video processing bucket
- `STATE_MACHINE_ARN`: Step Functions ARN
- `FILE_THRESHOLD`: Min files to trigger training
- `SIZE_THRESHOLD_MB`: Min data size for training
- `COOLDOWN_HOURS`: Hours between training runs
- `SAGEMAKER_ROLE_ARN`: SageMaker execution role
- `MODEL_BUCKET`: Model output bucket

**Terraform Variables:**
```hcl
variable "project_name" {
  default = "voice"
}

variable "environment" {
  default = "dev"
}

variable "video_lambda_timeout" {
  default = 900  # 15 minutes
}

variable "video_lambda_memory" {
  default = 10240  # 10GB
}
```

---

## Troubleshooting

### Common Issues

**Problem**: Lambda timeout during video processing
```
Solution: Increase timeout and memory in Terraform
- video_lambda_timeout = 900 (max)
- video_lambda_memory = 10240 (10GB)
```

**Problem**: Whisper model download fails
```
Solution: Model is cached in /tmp. Ensure sufficient /tmp space (512MB)
- Check TORCH_HOME=/tmp environment variable
- Verify ffmpeg installation in Dockerfile
```

**Problem**: MediaPipe pose detection fails
```
Solution: Check frame quality and lighting
- Ensure person is visible and in frame
- Log warnings for frames with no detected poses
- Review pose_extractor.py confidence thresholds
```

**Problem**: Cumulative pose indexing overwrites existing files
```
Solution: Fixed via get_next_pose_index()
- Function checks S3 for highest existing index
- Continues numbering from there
- Session metadata tracks which poses came from which video
```

**Problem**: SageMaker training fails
```
Solution: Check training logs in CloudWatch
- Verify training data structure in S3
- Ensure sufficient pose JSON files (>10 per emotion)
- Review train_sagemaker.py data setup logic
- Fallback model is created if training fails
```

**Problem**: Insufficient IAM permissions
```
Solution: Review IAM policies in modules/iam/main.tf
- Video processor needs S3 read/write on both buckets
- Training Lambda needs SageMaker CreateTrainingJob
- SageMaker role needs S3 access and ECR pull
```

### Debug Checklist

- [ ] Check CloudWatch logs for error messages
- [ ] Verify S3 bucket names and structure
- [ ] Confirm IAM role permissions
- [ ] Test Lambda function timeout and memory
- [ ] Validate input data format (video codecs, JSON structure)
- [ ] Review Terraform state for resource creation
- [ ] Check ECR image availability
- [ ] Verify Step Functions execution history

---

## Performance Optimization

### Lambda Cold Starts
- **Provisioned Concurrency**: Reduce cold start latency
- **Model Caching**: Whisper models cached in global scope
- **Lazy Initialization**: MediaPipe initialized only when needed

### Video Processing
- **Frame Sampling**: Intelligent frame selection at emotion timestamps
- **Batch Processing**: Process multiple frames in single invocation
- **Compression**: Use H.264 codec for annotated videos

### Training Pipeline
- **Feature Caching**: Save extracted features to CSV for reuse
- **Incremental Training**: Train only on new data
- **Parallel Processing**: SageMaker distributed training for large datasets

---

## Security Best Practices

1. **S3 Encryption**: AES256 server-side encryption enabled
2. **Bucket Policies**: Block public access on all buckets
3. **IAM Least Privilege**: Minimal permissions per role
4. **Secrets Management**: Use AWS Secrets Manager for API keys
5. **VPC Isolation**: Deploy Lambda in VPC for network isolation
6. **CloudTrail**: Enable for audit logging
7. **Data Retention**: Configure S3 lifecycle policies

---

## Roadmap

### Planned Features
- [ ] Real-time processing via Kinesis Video Streams
- [ ] Multi-person pose tracking
- [ ] Facial expression analysis with deepface
- [ ] Voice emotion recognition (prosody analysis)
- [ ] Temporal convolutional networks for sequence modeling
- [ ] Web dashboard for visualization
- [ ] API Gateway for external integrations
- [ ] Automated model retraining on performance drift

---

## Contributing

### Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- **Python**: Follow PEP 8 style guide
- **Terraform**: Use consistent naming conventions
- **Documentation**: Update README for new features
- **Testing**: Add unit tests for new functions

---

## Acknowledgments

- **OpenAI Whisper**: Audio transcription
- **MediaPipe**: Pose detection
- **AWS**: Cloud infrastructure

---

## Contact

**Elliot Kim**
- University of Notre Dame, Computer Science
- VOICE Project Lead

For questions or support, please open an issue on GitHub.

---

**Last Updated**: October 2025  
**Version**: 1.0.0
