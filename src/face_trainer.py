# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import os
import logging
from settings.settings import PATHS, FACE_DETECTION, LBPH_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_images_and_labels(path: str):
    """
    Load face images and corresponding labels from the given directory path.
    Returns:
        tuple: (face_samples, ids) Lists of face samples and corresponding labels.
    """
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
        faceSamples = []
        ids = []
        logger.info(f"Found {len(imagePaths)} image files in {path}")

        for imagePath in imagePaths:
            try:
                # Extract the user ID from the image file name
                fname = os.path.split(imagePath)[-1]
                id = int(fname.split('-')[1])
                # Load as grayscale (all images are already cropped faces)
                img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Could not read image: {imagePath}")
                    continue
                # Resize to fixed size for consistency
                try:
                    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    pass
                faceSamples.append(img)
                ids.append(id)
            except Exception as e:
                logger.warning(f"Skipping {imagePath}: {e}")
        logger.info(f"Loaded {len(faceSamples)} faces, {len(set(ids))} unique IDs")
        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition training...")
        
        # Initialize face recognizer with configurable params
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=LBPH_PARAMS.get('radius', 1),
                neighbors=LBPH_PARAMS.get('neighbors', 8),
                grid_x=LBPH_PARAMS.get('grid_x', 8),
                grid_y=LBPH_PARAMS.get('grid_y', 8)
            )
        except Exception:
            # Fallback if cv2 bindings have a different signature
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Get training data
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        
        if not faces or not ids:
            raise ValueError("No training data found")
            
        # Train the model
        logger.info("Training model...")
        recognizer.train(faces, np.array(ids))
        
        # Save the model
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained with {len(np.unique(ids))} faces")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
