# training_lambda/train.py - SESSION-BASED VERSION
"""
This version creates multiple training samples from pose directories
by using the .sessions metadata to split poses by video source
"""
import os
import sys
import numpy as np
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import argparse

from extract import PoseFeatureExtractor

# Constants
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Minimum samples required for training
MIN_SAMPLES_PER_CLASS = 3
MIN_TOTAL_SAMPLES = 10

def load_session_metadata(emotion_dir):
    """
    Load session metadata to map pose files to video sources
    
    Args:
        emotion_dir: Directory containing pose files and .sessions folder
        
    Returns:
        Dictionary mapping pose indices to session info
    """
    sessions_dir = os.path.join(emotion_dir, '.sessions')
    session_map = {}
    
    if not os.path.exists(sessions_dir):
        print(f"No .sessions directory found in {emotion_dir}")
        return None
    
    # Load all session files
    session_files = [f for f in os.listdir(sessions_dir) if f.endswith('_session.json')]
    
    for session_file in session_files:
        session_path = os.path.join(sessions_dir, session_file)
        try:
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            
            # Extract video file name (without extension) as session ID
            video_file = session_data.get('video_file', '')
            session_id = os.path.splitext(os.path.basename(video_file))[0]
            
            # Get pose index range for this session
            indices = session_data.get('pose_indices', {})
            start_idx = indices.get('start', 0)
            end_idx = indices.get('end', 0)
            
            # Map each pose index to this session
            for idx in range(start_idx, end_idx + 1):
                pose_filename = f"pose_{idx:06d}.json"
                session_map[pose_filename] = {
                    'session_id': session_id,
                    'video_file': video_file,
                    'emotion': session_data.get('emotion', 'unknown')
                }
        
        except Exception as e:
            print(f"Error loading session file {session_file}: {e}")
    
    return session_map

def group_poses_by_session(emotion_dir, session_map):
    """
    Group pose files by their source video session
    
    Args:
        emotion_dir: Directory containing pose files
        session_map: Mapping from pose files to sessions
        
    Returns:
        Dictionary of session_id -> list of pose file paths
    """
    if session_map is None:
        # Fallback: treat all poses as one session
        pose_files = [f for f in os.listdir(emotion_dir) 
                     if f.endswith('.json') and not f.startswith('.')]
        return {'default': [os.path.join(emotion_dir, f) for f in sorted(pose_files)]}
    
    sessions = {}
    
    # Group pose files by session
    for pose_file in os.listdir(emotion_dir):
        if not pose_file.endswith('.json') or pose_file.startswith('.'):
            continue
        
        # Check if this pose belongs to a tracked session
        if pose_file in session_map:
            session_info = session_map[pose_file]
            session_id = session_info['session_id']
            
            if session_id not in sessions:
                sessions[session_id] = []
            
            sessions[session_id].append(os.path.join(emotion_dir, pose_file))
        else:
            # Orphaned pose file (no session metadata)
            if 'untracked' not in sessions:
                sessions['untracked'] = []
            sessions['untracked'].append(os.path.join(emotion_dir, pose_file))
    
    # Sort pose files within each session
    for session_id in sessions:
        sessions[session_id].sort()
    
    return sessions

def find_emotion_directories(data_dir):
    """
    Find emotion directories (e.g., happiness, sadness)
    
    Args:
        data_dir: Root data directory
        
    Returns:
        List of (emotion_dir_path, emotion_name) tuples
    """
    emotion_dirs = []
    
    # Look for emotion directories directly under data_dir
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        
        # Skip hidden directories and files
        if item.startswith('.') or not os.path.isdir(item_path):
            continue
        
        # Check if directory name matches an emotion
        emotion = determine_emotion_from_path(item_path)
        if emotion:
            emotion_dirs.append((item_path, emotion))
    
    return emotion_dirs

def determine_emotion_from_path(path):
    """
    Determine emotion from directory or file path
    
    Args:
        path: Directory or file path
        
    Returns:
        Emotion name or None
    """
    path_lower = path.lower()
    
    # Check for emotion names in the path
    for emotion in EMOTIONS:
        if emotion.lower() in path_lower:
            return emotion
    
    return None

def extract_features_from_sessions(emotion_dirs, feature_extractor):
    """
    Extract features from pose data grouped by video session
    Creates one training sample per video session
    
    Args:
        emotion_dirs: List of (emotion_dir_path, emotion_name) tuples
        feature_extractor: PoseFeatureExtractor instance
        
    Returns:
        Features, labels, and session info
    """
    features = []
    labels = []
    session_info = []
    
    print(f"Processing {len(emotion_dirs)} emotion directories...")
    
    for emotion_dir, emotion in emotion_dirs:
        print(f"\nProcessing {emotion} from {emotion_dir}")
        
        # Load session metadata
        session_map = load_session_metadata(emotion_dir)
        
        # Group poses by session
        sessions = group_poses_by_session(emotion_dir, session_map)
        
        print(f"Found {len(sessions)} sessions with poses:")
        for session_id, pose_files in sessions.items():
            print(f"  - {session_id}: {len(pose_files)} poses")
        
        # Extract features for each session separately
        for session_id, pose_files in tqdm(sessions.items(), desc=f"Extracting {emotion}"):
            if len(pose_files) < 5:  # Skip sessions with very few poses
                print(f"  Skipping {session_id}: only {len(pose_files)} poses (need at least 5)")
                continue
            
            try:
                # Extract features from this session's poses
                session_features = extract_features_from_pose_files(pose_files, feature_extractor)
                
                if session_features is not None and len(session_features) > 0:
                    features.append(session_features)
                    labels.append(emotion)
                    session_info.append({
                        'session_id': session_id,
                        'emotion': emotion,
                        'num_poses': len(pose_files)
                    })
            
            except Exception as e:
                print(f"  Error extracting features from session {session_id}: {e}")
    
    if not features:
        print("\n❌ No valid features extracted from any session")
        return None, None, None
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"\n📊 Feature Extraction Summary:")
    print(f"Extracted {features.shape[1]} features from {len(labels)} video sessions")
    
    # Print session distribution
    emotion_counts = pd.Series(labels).value_counts()
    print(f"\nSessions per emotion:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} sessions")
    
    # Print detailed session info
    print(f"\nDetailed session breakdown:")
    for info in session_info:
        print(f"  {info['session_id']} ({info['emotion']}): {info['num_poses']} poses")
    
    return features, labels, session_info

def extract_features_from_pose_files(pose_files, feature_extractor):
    """
    Extract features from a list of pose JSON files
    
    Args:
        pose_files: List of pose file paths
        feature_extractor: PoseFeatureExtractor instance
        
    Returns:
        Extracted features
    """
    pose_frames = []
    
    for pose_file in pose_files:
        try:
            with open(pose_file, 'r') as f:
                data = json.load(f)
            
            # Extract keypoints (OpenPose format)
            keypoints = None
            
            if 'people' in data and len(data['people']) > 0:
                if 'pose_keypoints_2d' in data['people'][0]:
                    flat_keypoints = data['people'][0]['pose_keypoints_2d']
                    keypoints = np.array(flat_keypoints).reshape(-1, 3)  # [keypoints, 3]
            
            if keypoints is not None:
                pose_frames.append(keypoints)
        
        except Exception as e:
            print(f"Error loading {pose_file}: {e}")
    
    if not pose_frames:
        return None
    
    # Convert to numpy array
    pose_frames = np.array(pose_frames)
    
    # Extract features from the sequence of poses
    features = feature_extractor.extract_features_from_pose_sequence(pose_frames)
    
    return features

def validate_training_data(labels):
    """
    Validate that we have enough data to train
    
    Args:
        labels: Array of emotion labels
        
    Returns:
        (is_valid, error_message)
    """
    if labels is None or len(labels) == 0:
        return False, "No training data available"
    
    # Check total samples
    if len(labels) < MIN_TOTAL_SAMPLES:
        return False, f"Need at least {MIN_TOTAL_SAMPLES} total samples (video sessions), got {len(labels)}"
    
    # Check samples per class
    emotion_counts = pd.Series(labels).value_counts()
    min_class_count = emotion_counts.min()
    
    if min_class_count < MIN_SAMPLES_PER_CLASS:
        insufficient_classes = [
            f"{emotion} ({count})" 
            for emotion, count in emotion_counts.items() 
            if count < MIN_SAMPLES_PER_CLASS
        ]
        return False, f"Need at least {MIN_SAMPLES_PER_CLASS} samples per class. Insufficient: {', '.join(insufficient_classes)}"
    
    return True, None

def train_emotion_model(features, labels, session_info, model_path, output_dir=None):
    """
    Train an emotion classification model
    
    Args:
        features: Extracted features
        labels: Emotion labels
        session_info: Information about training sessions
        model_path: Path to save the model
        output_dir: Directory to save visualizations
        
    Returns:
        Trained model, scaler, and label encoder
    """
    if features is None or labels is None:
        print("Error: Features or labels are None")
        return None, None, None
    
    # Validate training data
    is_valid, error_msg = validate_training_data(labels)
    if not is_valid:
        print(f"\n❌ Training data validation failed: {error_msg}")
        print("\nYou need to upload more videos. Each video becomes one training sample.")
        print(f"Current: {len(labels)} video sessions")
        print(f"Required: {MIN_TOTAL_SAMPLES} total, {MIN_SAMPLES_PER_CLASS} per emotion")
        return None, None, None
    
    try:
        print("\n✅ Training data validation passed!")
        print("Training emotion model...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        print(f"Label mapping: {dict(enumerate(label_encoder.classes_))}")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels_encoded
        )
        
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
        
        # Cross-validation
        print("Performing cross-validation...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(np.unique(y_train))))
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"\n📊 Model Performance:")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        
        # Classification report
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
        
        # Save visualizations
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
            
            # Feature importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump((model, scaler, label_encoder), f)
        
        # Save session info
        session_info_path = os.path.join(output_dir or os.path.dirname(model_path), 'training_sessions.json')
        with open(session_info_path, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        print(f"\n✅ Model saved to {model_path}")
        print(f"✅ Session info saved to {session_info_path}")
        
        return model, scaler, label_encoder
    
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model with session-based samples')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing emotion subdirectories')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_path = os.path.join(args.output_dir, 'advanced_emotion_model.pkl')
    
    print("="*60)
    print("SESSION-BASED EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize feature extractor
    feature_extractor = PoseFeatureExtractor()
    
    # Find emotion directories
    emotion_dirs = find_emotion_directories(args.data_dir)
    
    if not emotion_dirs:
        print(f"❌ No emotion directories found in {args.data_dir}")
        sys.exit(1)
    
    print(f"Found {len(emotion_dirs)} emotion directories:")
    for dir_path, emotion in emotion_dirs:
        print(f"  - {emotion}: {dir_path}")
    print()
    
    # Extract features (one sample per video session)
    features, labels, session_info = extract_features_from_sessions(emotion_dirs, feature_extractor)
    
    # Train model
    if features is not None and labels is not None:
        result = train_emotion_model(features, labels, session_info, model_path, args.output_dir)
        
        if result == (None, None, None):
            print("\n❌ Training failed")
            print("\n💡 Tip: Upload more videos. Each video = 1 training sample")
            print(f"   Need: {MIN_TOTAL_SAMPLES} total videos, {MIN_SAMPLES_PER_CLASS} per emotion")
            sys.exit(1)
        else:
            print("\n🎉 Training completed successfully!")
    else:
        print("❌ Could not extract features")
        sys.exit(1)

if __name__ == "__main__":
    main()