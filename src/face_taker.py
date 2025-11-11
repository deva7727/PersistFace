import cv2
import os
import json
import time
from settings.settings import *

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_camera(source):
    try:
        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source), CAMERA.get('backend', cv2.CAP_ANY))
            if not cap.isOpened():
                return None
            return cap
        # For network streams, just hand off to VideoCapture but allow caller to validate first frame
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
        # Reduce buffer for lower latency when possible
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    except Exception:
        return None


def _build_auth_url(url, username, password):
    if not username or not password:
        return url
    if '://' in url:
        proto, rest = url.split('://', 1)
        return f"{proto}://{username}:{password}@{rest}"
    return f"http://{username}:{password}@{url}"


def _probe_candidates(base_url, username=None, password=None, timeout=3):
    """
    Generate common IP webcam candidate endpoints based on a base URL.
    """
    u = base_url.rstrip('/')
    candidates = [
        u,
        u + '/video',
        u + '/video_feed',
        u + '/mjpeg',
        u + '/stream',
        u + '/shot.jpg',
        u + '/snapshot.jpg',
        u + '/?action=stream',
        u + '/cgi-bin/stream.cgi',
        u + '/video.cgi',
    ]
    # If provided credentials, add same candidates with auth embedded
    if username and password:
        return [_build_auth_url(c, username, password) for c in candidates]
    return candidates


def find_working_ip_camera(url, username=None, password=None, attempts=3):
    """
    Try multiple common IP-camera endpoints and return a working cv2.VideoCapture
    that yields at least one frame, or None if none work.
    """
    import urllib.request
    candidates = _probe_candidates(url, username, password)
    for candidate in candidates:
        try:
            # Quick network probe: try to open URL headers (non-blocking small timeout)
            req = urllib.request.Request(candidate, method='HEAD')
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:
                    # Accept if content-type looks like video or jpeg
                    ctype = resp.headers.get('Content-Type', '')
            except Exception:
                # HEAD may not be allowed; fall back to trying VideoCapture directly
                ctype = ''
            # Try opening with VideoCapture and read a frame
            cap = initialize_camera(candidate)
            if cap is None:
                continue
            ok = False
            for _ in range(attempts):
                ret, frame = cap.read()
                if ret and frame is not None:
                    ok = True
                    break
                time.sleep(0.2)
            if ok:
                return cap
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            continue
    return None

def get_face_id(image_dir):
    # Find max ID in image filenames
    max_id = 0
    for fname in os.listdir(image_dir):
        if fname.startswith('Users-') and fname.endswith('.jpg'):
            try:
                parts = fname.split('-')
                id_part = parts[1]
                max_id = max(max_id, int(id_part))
            except Exception:
                continue
    return max_id + 1

def save_name(face_id, face_name, names_file):
    names = {}
    if os.path.exists(names_file):
        try:
            with open(names_file, 'r') as f:
                names = json.load(f)
        except Exception:
            names = {}
    names[str(face_id)] = face_name
    with open(names_file, 'w') as f:
        json.dump(names, f)
def main():
    cam = None
    try:
        # Initialize
        create_directory(PATHS['image_dir'])
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            print("Error loading cascade classifier")
            return
        try:
            detector = cv2.FaceDetectorYN.create(
                PATHS['yunet_model'], "", (CAMERA['process_width'], int(CAMERA['process_width'] * 3/4)), 0.9, 0.3, 5000)
        except Exception:
            detector = None
        print("\nSelect camera source:")
        print("1. Laptop/USB Camera")
        print("2. IP Camera")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ").strip()
        if choice == '3':
            print("Exiting...")
            return
        camera_source = '0'
        if choice == '2':
            ip_url = input("Enter IP camera URL (e.g., http://10.10.10.245:8080/video): ").strip()
            username = input("Enter username (leave blank if none): ").strip()
            password = input("Enter password (leave blank if none): ").strip()
        elif choice == '1':
            cam_index = input("Enter camera index (0 for default, 1 for external): ").strip() or '0'
            camera_source = cam_index
        else:
            print("Invalid choice. Please try again.")
            return
        # Initialize camera connection: for IP cameras try probing several common endpoints
        if choice == '2':
            cam = find_working_ip_camera(ip_url, username or None, password or None)
            if cam is None:
                print("Failed to connect to IP camera. Tried common endpoints. Check stream URL and app settings.")
                return
        else:
            cam = initialize_camera(camera_source)
            if cam is None:
                print("Failed to connect to camera. Please check the camera source and try again.")
                return
        face_name = input('\nEnter user name and press <return> -->  ').strip()
        if not face_name:
            print("Name cannot be empty")
            return
        face_id = get_face_id(PATHS['image_dir'])
        save_name(face_id, face_name, PATHS['names_file'])
        print(f"Capturing faces for {face_name} (ID: {face_id})")
        count = 0
        while count < TRAINING['samples_needed']:
            ret, img = cam.read()
            if not ret:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = []
            if detector is not None:
                try:
                    faces_array = detector.detect(img)
                    if faces_array is not None and len(faces_array) > 0:
                        for face in faces_array:
                            x, y, w, h = int(face[0]), int(face[1]), int(face[2] - face[0]), int(face[3] - face[1])
                            faces.append((x, y, w, h))
                except Exception:
                    pass
            if not faces:
                faces_c = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=FACE_DETECTION['scale_factor'],
                    minNeighbors=FACE_DETECTION['min_neighbors'],
                    minSize=FACE_DETECTION['min_size'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                faces = list(faces_c)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                try:
                    face_img = cv2.resize(face_img, (200, 200), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    continue
                img_path = f'{PATHS["image_dir"]}/Users-{face_id}-{count+1}.jpg'
                cv2.imwrite(img_path, face_img)
                print(f"Saved {img_path}")
                if TRAINING.get('augment', False):
                    for scale in (0.9, 1.1):
                        try:
                            aug = cv2.resize(face_img, (int(200*scale), int(200*scale)), interpolation=cv2.INTER_LINEAR)
                            aug = cv2.resize(aug, (200, 200), interpolation=cv2.INTER_LINEAR)
                            aug_path = f'{PATHS["image_dir"]}/Users-{face_id}-{count+1}-aug{int(scale*10)}.jpg'
                            cv2.imwrite(aug_path, aug)
                        except Exception:
                            continue
                    if TRAINING.get('augment_flip', False):
                        try:
                            flip = cv2.flip(face_img, 1)
                            flip_path = f'{PATHS["image_dir"]}/Users-{face_id}-{count+1}-flip.jpg'
                            cv2.imwrite(flip_path, flip)
                        except Exception:
                            continue
                count += 1
            cv2.imshow('Face Capture', img)
            if cv2.waitKey(100) & 0xff == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        print(f"Done. {count} faces captured.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
