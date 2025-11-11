"""
Configuration settings for the face recognition system
"""
import os
import cv2
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force XCB backend for Qt

# Camera settings
CAMERA = {
    'index': 0,  # Default camera (0 is usually built-in webcam)
    'width': 640,
    'height': 480,
    'fps': 30,   # Target FPS
    'ip_camera': {
        'width': 480,  # Smaller resolution for IP camera
        'height': 360,
        'fps': 15      # Lower FPS for IP camera to reduce network load
    },
    'backend': cv2.CAP_V4L2,  # Use V4L2 backend for better performance on Linux
}

# Processing options: width to resize frames for detection/recognition.
# Increasing this improves detection of distant faces but costs CPU.
CAMERA['process_width'] = 480

# Face detection settings
FACE_DETECTION = {
    'scale_factor': 1.3,
    'min_neighbors': 5,
    'min_size': (30, 30),
    # LBPH recognizer distance threshold (lower is better). Tune if you see false positives/negatives.
    'recognition_threshold': 80
}

# Training settings. Number of images needed to train the model.
TRAINING = {
    'samples_needed': 20
}

# When True, augment captured training images with small scale changes and flips
TRAINING['augment'] = True
TRAINING['augment_flip'] = True

# LBPH recognizer parameters (used during training)
LBPH_PARAMS = {
    'radius': 1,
    'neighbors': 8,
    'grid_x': 8,
    'grid_y': 8
}

# Recognition smoothing / stability settings
RECOGNITION = {
    # How many recent predictions to keep for voting
    'history_len': 5,
    # Minimum votes (in history window) required to accept a label
    'confirm_votes': 3,
    # Time (seconds) after which a tracked face is forgotten if not seen
    'forget_timeout': 1.5,
    # Bounding box smoothing factor (0..1) where closer to 0 = more smoothing
    'bbox_smoothing_alpha': 0.6
}

# training dependencies
PATHS = {
    'image_dir': 'images',
    'cascade_file': 'haarcascade_frontalface_default.xml',
    'profile_cascade_file': 'haarcascade_profileface.xml',
    'yunet_model': 'yunet.onnx',
    'names_file': 'names.json',
    'trainer_file': 'trainer.yml'
}
