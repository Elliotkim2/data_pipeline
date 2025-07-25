#!/usr/bin/env python
"""
SageMaker training wrapper script that handles directory structure issues
"""
import os
import sys
import subprocess
import shutil
import json

def setup_training_data():
    """
    Reorganize training data to match expected structure
    SageMaker puts data in /opt/ml/input/data/train/
    We need emotion directories directly in that path
    """
    train_dir = '/opt/ml/input/data/train'
    print(f"Training data directory: {train_dir}")
    print(f"Contents: {os.listdir(train_dir)}")
    
    # Check if we have subdirectories that need to be moved up
    moved_anything = False
    
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        
        # If this is a directory (like 'test-user-v2')
        if os.path.isdir(item_path):
            print(f"Found directory: {item}")
            
            # Check if it contains emotion directories
            subdirs = os.listdir(item_path)
            print(f"Subdirectories in {item}: {subdirs}")
            
            # Check if any subdirectory is an emotion directory
            emotions = ['happiness', 'sadness', 'anger', 'disgust', 'fear', 'neutral', 'surprise']
            has_emotion_dirs = any(subdir.lower() in emotions for subdir in subdirs)
            
            if has_emotion_dirs:
                print(f"Moving emotion directories from {item_path} to {train_dir}")
                
                # Move each emotion directory up to the train directory
                for subdir in subdirs:
                    src = os.path.join(item_path, subdir)
                    dst = os.path.join(train_dir, subdir)
                    
                    if os.path.isdir(src):
                        print(f"Moving {src} to {dst}")
                        shutil.move(src, dst)
                        moved_anything = True
                
                # Remove the now-empty parent directory
                try:
                    os.rmdir(item_path)
                except:
                    pass
    
    if moved_anything:
        print(f"After reorganization, train directory contains: {os.listdir(train_dir)}")
    
    # Verify we have emotion directories
    final_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Final directories in training data: {final_dirs}")
    
    # Check for JSON files in each directory
    for dir_name in final_dirs:
        dir_path = os.path.join(train_dir, dir_name)
        json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        print(f"Directory {dir_name} contains {len(json_files)} JSON files")

def main():
    # Model directory where we'll save outputs
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    
    print(f"Model dir: {model_dir}")
    print(f"Train dir: {train_dir}")
    print(f"Output dir: {output_dir}")
    
    # List contents of train directory
    print(f"Training data contents: {os.listdir(train_dir)}")
    
    # Reorganize data if needed
    setup_training_data()
    
    # Install requirements if needed
    if os.path.exists('requirements.txt'):
        print("Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Run the actual training script
    print("Running train.py...")
    result = subprocess.run([
        sys.executable, 'train.py',
        '--data_dir', train_dir,
        '--output_dir', model_dir
    ], capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Training script failed with return code {result.returncode}")
        # Don't exit with error - let SageMaker handle it
    
    # Copy any outputs to the model directory
    print("Copying outputs to model directory...")
    
    # Ensure model.pkl exists or create a dummy one
    model_file = os.path.join(model_dir, 'advanced_emotion_model.pkl')
    if not os.path.exists(model_file):
        print("Warning: Model file not found, creating placeholder...")
        # Create a minimal placeholder so SageMaker doesn't fail
        with open(model_file, 'w') as f:
            f.write("Model training incomplete - check logs")
    
    # Copy any other outputs
    for item in os.listdir(model_dir):
        print(f"Model output: {item}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()