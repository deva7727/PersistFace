# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import time
import logging
from collections import deque, Counter
from settings.settings import CAMERA, FACE_DETECTION, PATHS
from settings.settings import RECOGNITION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_camera(source: str = '0') -> cv2.VideoCapture:
    """
    Initialize the camera with error handling and optimization
    
    Parameters:
        source (str): Camera source - can be device index (e.g. '0') or IP camera URL
    Returns:
        cv2.VideoCapture: Initialized camera object
    """
    try:
        is_ip_camera = False
        # Convert to integer if source is a number (local camera)
        if source.isdigit():
            source = int(source)
        elif source.startswith('http'):
            is_ip_camera = True
            # For IP cameras, append video feed endpoint if not specified
            if not source.endswith(('/video', '/shot.jpg', '/video_feed')):
                source = source.rstrip('/') + '/video'
                
        # Set OpenCV options for better streaming performance
        if is_ip_camera:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            
        cam = cv2.VideoCapture(source)
        if not cam.isOpened():
            logger.error(f"Could not open camera source: {source}")
            return None
            
        # Configure camera properties
        if isinstance(source, int):
            # Settings for local camera
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            cam.set(cv2.CAP_PROP_FPS, CAMERA['fps'])
        else:
            # Optimize settings for IP camera
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for lowest latency
            cam.set(cv2.CAP_PROP_FPS, CAMERA['ip_camera']['fps'])
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['ip_camera']['width'])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['ip_camera']['height'])
            # Additional optimizations for IP camera
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
        # Test camera read and get actual frame size
        ret, frame = cam.read()
        if not ret:
            logger.error(f"Could not read from camera source: {source}")
            cam.release()
            return None
            
        if is_ip_camera:
            # Log actual frame size and FPS for IP camera
            actual_fps = cam.get(cv2.CAP_PROP_FPS)
            actual_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"IP Camera connected - Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
                
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        if 'cam' in locals():
            cam.release()
        return None

def load_names(filename: str) -> dict:
    """
    Load name mappings from JSON file
    
    Parameters:
        filename (str): Path to the JSON file containing name mappings
    Returns:
        dict: Dictionary mapping IDs to names
    """
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

if __name__ == "__main__":
        cam = None
        logger.info("Starting face recognition system...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        recognizer.read(PATHS['trainer_file'])
        
        # Load face detector (YuNet DNN for better accuracy)
        detector = cv2.FaceDetectorYN.create(
            PATHS['yunet_model'], 
            "", 
            (CAMERA['process_width'], int(CAMERA['process_width'] * 3/4)),  # input size
            0.9,  # score threshold
            0.3,  # nms threshold
            5000  # top_k
        )
        if detector is None:
            raise ValueError("Failed to load YuNet face detector")
        
        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")
        
        while True:
            # Ask for camera source
            print("\nSelect camera source:")
            print("1. Laptop/USB Camera")
            print("2. IP Camera")
            print("3. Exit")
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == '3':
                print("Exiting...")
                break
                
            camera_source = '0'  # default to laptop camera
            if choice == '2':
                camera_source = input("Enter IP camera URL (e.g., http://10.10.10.245:8080): ").strip()
            elif choice == '1':
                cam_index = input("Enter camera index (0 for default, 1 for external): ").strip() or '0'
                camera_source = cam_index
            else:
                print("Invalid choice. Please try again.")
                continue
            
            cam = initialize_camera(camera_source)
            if cam is None:
                print("Failed to connect to camera. Please check the camera source and try again.")
                continue
            else:
                break
                
        if cam is None:
            raise ValueError("No camera selected or failed to initialize camera")
            
        logger.info("Press 'ESC' to exit.")
        
            # 🎯 NOTE: Agar "Skipped" frames aa rahe hain, toh settings.py mein 'fps' ko kam (e.g., 15) karein.
        frame_interval = 1.0 / CAMERA.get('fps', 30)  # Time between frames
        last_frame_time = time.time()
        frames_count = 0
        fps = 0
        skip_count = 0
        face_detector = cv2.CascadeClassifier(PATHS['cascade_file'])

        # Recognition tracking state: small IoU-based tracker to stabilize detections
        # tracks: id -> { 'votes': deque, 'confirmed_name': str|None, 'last_top': str|None,
        #                  'candidate_count': int, 'bbox': (x,y,w,h), 'last_seen': ts }
        tracks = {}
        next_track_id = 1
        # recently removed tracks (for re-attachment by appearance/name)
        removed_tracks = {}
        history_len = RECOGNITION.get('history_len', 5)
        confirm_votes = RECOGNITION.get('confirm_votes', 3)
        forget_timeout = RECOGNITION.get('forget_timeout', 1.5)
        # bbox_smoothing: weight for old bbox (0..1). Higher => more smoothing.
        bbox_smoothing = RECOGNITION.get('bbox_smoothing_alpha', 0.6)
        # How many frames to predict forward when a detection is briefly lost
        max_predict = max(1, int(forget_timeout * CAMERA.get('fps', 15)))
        # locked tracks persist longer and resist being re-assigned
        # Reduced timeout so box doesn't stick to empty area when user moves away
        forget_timeout_locked = RECOGNITION.get('forget_timeout_locked', forget_timeout * 3)
        hist_threshold = RECOGNITION.get('hist_threshold', 0.6)  # stricter for better accuracy
        # grace frames for locked tracks (allow brief occlusion/pose change without removal)
        locked_grace_seconds = RECOGNITION.get('locked_grace_seconds', 2.5)
        locked_grace_frames = max(1, int(locked_grace_seconds * CAMERA.get('fps', 15)))
        # minimum area (pixels) to create a new track for an Unknown detection
        min_area_for_unknown_track = RECOGNITION.get('min_area_for_unknown_track', 2000)

        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
            yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            denom = float(boxAArea + boxBArea - interArea)
            return interArea / denom if denom > 0 else 0.0

        # Appearance descriptor: color histogram in HSV space for simple re-ID
        def compute_color_hist(bgr_roi):
            try:
                if bgr_roi is None or bgr_roi.size == 0:
                    return None
                hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
                # use H and S channels for robustness to lighting
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                return hist
            except Exception:
                return None
        # CLAHE for contrast limited histogram equalization (better than global equalize)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        except Exception:
            clahe = None
        process_this_frame = True
        
        while True:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Read frame without processing if we're behind
            ret = cam.grab()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
                
            # Only process every second frame if we're falling behind
            # If we're falling behind, skip occasional frames — be conservative to avoid
            # excessive skipping which increases latency. Increase threshold so skipping
            # happens only when we're significantly behind.
            if elapsed > frame_interval * 3:
                skip_count += 1
                if skip_count % 2 != 0:  # Skip every other frame
                    continue
                    
            # Retrieve and process frame
            ret, img = cam.retrieve()
            if not ret:
                logger.warning("Failed to retrieve frame")
                continue

            # NOTE: frame size will be defined AFTER any resizing below to keep coordinates consistent
            frames_count += 1
            if elapsed >= 1.0:  # Update FPS every second
                fps = frames_count / elapsed
                frames_count = 0
                last_frame_time = current_time
            
            # Skip frames if we're falling behind
            if cam.get(cv2.CAP_PROP_POS_FRAMES) > 1:
                cam.grab()  # Skip frames to catch up
                
            # Resize frame for faster processing - use configured process width
            target_width = CAMERA.get('process_width', 320)  # Larger gives better distant face detection
            # Use current dimensions from img (freshly retrieved)
            orig_h, orig_w = img.shape[:2]
            if orig_w > target_width:
                scale = target_width / orig_w
                img = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

            # Now set the frame size based on the (possibly) resized image so all coords match
            frame_height, frame_width = img.shape[:2]

            # Ensure YuNet input size matches the current frame size. YuNet requires
            # setInputSize(width,height) equal to the image size or it will raise an error.
            try:
                detector.setInputSize((frame_width, frame_height))
            except Exception:
                # Some OpenCV builds may require width,height as integers in a tuple
                try:
                    detector.setInputSize((int(frame_width), int(frame_height)))
                except Exception:
                    # If we can't set input size, continue without detection (fallback to cascade below)
                    logger.debug("Could not set YuNet input size; will fallback to cascade if needed")
            
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if clahe is not None:
                try:
                    gray = clahe.apply(gray)
                except Exception:
                    gray = cv2.equalizeHist(gray)
            else:
                gray = cv2.equalizeHist(gray)
            
            # Detect faces using YuNet DNN (preferred). If YuNet fails due to input size
            # mismatch or other issues, we'll catch and fall back to Haar cascade below.
            faces_array = np.zeros((100, 15), dtype=np.float32)
            try:
                detector.detect(img, faces_array)
            except Exception as e:
                logger.debug(f"YuNet detect failed (will fallback to cascade): {e}")
                faces_array = np.zeros((0, 15), dtype=np.float32)

            # If YuNet found nothing, try detection on a horizontally flipped image
            # (helps with profile faces and some camera mirrors). Map the coords back.
            faces_from_yunet = []
            for i in range(faces_array.shape[0]):
                face = faces_array[i]
                if face[0] == 0 and face[1] == 0:
                    break
                x, y, w, h = int(face[0]), int(face[1]), int(face[2] - face[0]), int(face[3] - face[1])
                faces_from_yunet.append((x, y, w, h))

            if not faces_from_yunet:
                try:
                    flipped = cv2.flip(img, 1)
                    faces_array_f = np.zeros((100, 15), dtype=np.float32)
                    try:
                        detector.detect(flipped, faces_array_f)
                    except Exception:
                        faces_array_f = np.zeros((0, 15), dtype=np.float32)
                    for i in range(faces_array_f.shape[0]):
                        face = faces_array_f[i]
                        if face[0] == 0 and face[1] == 0:
                            break
                        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2] - face[0]), int(face[3] - face[1])
                        # map x coordinate from flipped image back to original
                        mx = frame_width - (fx + fw)
                        faces_from_yunet.append((mx, fy, fw, fh))
                    if faces_from_yunet:
                        logger.debug(f"YuNet found faces on flipped image: {faces_from_yunet}")
                except Exception:
                    pass

            faces = faces_from_yunet

            # If YuNet didn't find any faces, fallback to Haar cascade detection
            if not faces:
                try:
                    faces_c = face_detector.detectMultiScale(
                        gray,
                        scaleFactor=FACE_DETECTION.get('scale_factor', 1.1),
                        minNeighbors=FACE_DETECTION.get('min_neighbors', 5),
                        minSize=FACE_DETECTION.get('min_size', (30, 30)),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    faces = list(faces_c)
                    if faces:
                        logger.debug(f"Cascade fallback detected faces: {faces}")
                except Exception as e:
                    logger.debug(f"Cascade detection failed: {e}")
            
            # Filter out invalid detections
            valid_faces = []
            for (fx, fy, fw, fh) in faces:
                if fw > 0 and fh > 0 and fx >= 0 and fy >= 0 and fx + fw <= frame_width and fy + fh <= frame_height:
                    valid_faces.append((fx, fy, fw, fh))
            faces = valid_faces
            
            # Process faces and apply IoU-based tracking + vote-based confirmation
            current_ts = time.time()
            matched_track_ids = set()

            # For each detected face, recognize and assign to an existing track (by IoU)
            for (x, y, w, h) in faces:
                # Extract face region for recognition
                face_roi = gray[y:y+h, x:x+w]
                if face_roi is None or face_roi.size == 0:
                    continue
                try:
                    face_for_recog = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    continue
                # Run the recognizer on the cropped and resized face ROI
                try:
                    pred_id, confidence = recognizer.predict(face_for_recog)
                except Exception as e:
                    logger.warning(f"Recognition error for ROI: {e}")
                    pred_id, confidence = -1, 9999

                # detection area (used to filter tiny Unknown detections)
                area = w * h

                threshold = FACE_DETECTION.get('recognition_threshold', 80)
                candidate_name = names.get(str(pred_id), "Unknown") if confidence < threshold else "Unknown"

                # Quick validation to reduce false-positive 'Unknown' boxes:
                # If recognizer says Unknown, re
                if candidate_name == "Unknown":
                    try:
                        # run a quick Haar check inside the ROI
                        haar_hits = face_detector.detectMultiScale(
                            face_roi,
                            scaleFactor=1.05,
                            minNeighbors=3,
                            minSize=(max(20, int(w * 0.35)), max(20, int(h * 0.35)))
                        )
                    except Exception:
                        haar_hits = ()
                    if (not hasattr(haar_hits, '__len__') or len(haar_hits) == 0) and area < min_area_for_unknown_track:
                        # likely a false positive; skip
                        continue

                # Precompute appearance hist for this detection (helps when face rotates)
                try:
                    color_roi_now = img[y:y+h, x:x+w]
                    now_hist = compute_color_hist(color_roi_now)
                except Exception:
                    now_hist = None

                # Helper: expand bbox by factor (centered)
                def expand_bbox(bbox, factor=1.3):
                    bx, by, bw0, bh0 = bbox
                    cx = bx + bw0/2
                    cy = by + bh0/2
                    nw = bw0 * factor
                    nh = bh0 * factor
                    nx = int(cx - nw/2)
                    ny = int(cy - nh/2)
                    return (nx, ny, int(nw), int(nh))

                # Match detection to existing tracks by IoU and appearance
                best_tid = None
                best_score = 0.0
                for tid, tr in tracks.items():
                    tbbox = tr.get('bbox', (0,0,0,0))
                    # for locked tracks, expand comparison bbox moderately to tolerate pose changes
                    comp_bbox = expand_bbox(tbbox, factor=1.6) if tr.get('locked') else tbbox
                    i = iou(comp_bbox, (x, y, w, h))
                    score = i
                    # if appearance histograms exist, use histogram similarity to boost score
                    tr_hist = tr.get('hist')
                    if tr_hist is not None and now_hist is not None:
                        try:
                            dist = cv2.compareHist(tr_hist, now_hist, cv2.HISTCMP_BHATTACHARYYA)
                            # convert to similarity (1 - dist) and weigh it
                            sim = max(0.0, 1.0 - dist)
                            weight = 0.7 if tr.get('locked') else 0.6
                            score = max(score, weight * sim + (1-weight) * score)
                        except Exception:
                            pass
                    # if locked and reasonable overlap, accept with moderate boost
                    if tr.get('locked') and i >= 0.1:
                        score = max(score, 0.28)
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                # Threshold for assigning to existing track (use best_score from matching)
                if best_score >= 0.25 and best_tid is not None:
                    tr = tracks[best_tid]
                    tr['last_seen'] = current_ts
                    tr['last_conf'] = confidence
                    # reset skipped frames
                    tr['skipped'] = 0
                    # If we have a concrete name, use it for voting/confirmation.
                    if candidate_name != "Unknown":
                        tr['votes'].append(candidate_name)
                        # voting and confirmation logic
                        most_common = Counter(tr['votes']).most_common(1)
                        top_name = most_common[0][0] if most_common else None
                        if top_name == tr.get('last_top'):
                            tr['candidate_count'] = tr.get('candidate_count', 0) + 1
                        else:
                            tr['last_top'] = top_name
                            tr['candidate_count'] = 1
                        if tr['candidate_count'] >= confirm_votes:
                            # Confirmed identity -> lock the track to this person IMMEDIATELY
                            if tr.get('confirmed_name') != top_name:
                                tr['confirmed_name'] = top_name
                                tr['locked'] = True
                                tr['lock_time'] = current_ts
                                # give locked tracks a long grace period (in frames) before they start aging out
                                tr['grace_left'] = locked_grace_frames
                                # capture appearance histogram for re-identification
                                try:
                                    bx, by, bw, bh = tr.get('bbox', (x, y, w, h))
                                    color_roi = img[by:by+bh, bx:bx+bw]
                                    tr['hist'] = compute_color_hist(color_roi)
                                except Exception:
                                    tr['hist'] = None
                        # Even if already confirmed, refresh lock and grace whenever seen
                        if tr.get('confirmed_name') and tr.get('confirmed_name') == top_name:
                            tr['locked'] = True
                            tr['grace_left'] = locked_grace_frames
                    else:
                        # Unknown detection: don't use as a vote (prevents locking on Unknown)
                        # but refresh grace if this is a locked track so it doesn't age out.
                        if tr.get('locked'):
                            tr['grace_left'] = locked_grace_frames
                    # Update Kalman filter with measured center and smooth bbox
                    try:
                        cx = int(x + w/2)
                        cy = int(y + h/2)
                        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
                        # correct then predict to get smoothed state
                        tr['kf'].correct(meas)
                        pred = tr['kf'].predict()
                        pcx, pcy = int(pred[0][0]), int(pred[1][0])
                        # map predicted center back to bbox keeping width/height
                        sw, sh = w, h
                        sx = int(pcx - sw/2)
                        sy = int(pcy - sh/2)
                        tr['bbox'] = (sx, sy, sw, sh)
                    except Exception:
                        tr['bbox'] = (x, y, w, h)
                    tracks[best_tid] = tr
                    matched_track_ids.add(best_tid)
                else:
                    # Try to re-attach to a recently removed track by name (helps when person briefly leaves)
                    reused_tid = None
                    if candidate_name != "Unknown":
                        reattach_window = max(2 * forget_timeout, 3.0)
                        # compute current detection's appearance hist for stronger matching
                        try:
                            color_roi_now = img[y:y+h, x:x+w]
                            now_hist = compute_color_hist(color_roi_now)
                        except Exception:
                            now_hist = None
                        for rid, rt in list(removed_tracks.items()):
                            # match on confirmed name or last_top vote
                            name_match = (rt.get('confirmed_name') == candidate_name) or (rt.get('last_top') == candidate_name)
                            if not name_match:
                                continue
                            if (current_ts - rt.get('last_seen', 0)) > reattach_window:
                                continue
                            # if both have histograms, use histogram distance to validate
                            rt_hist = rt.get('hist')
                            if rt_hist is not None and now_hist is not None:
                                try:
                                    dist = cv2.compareHist(rt_hist, now_hist, cv2.HISTCMP_BHATTACHARYYA)
                                except Exception:
                                    dist = 1.0
                                if dist <= hist_threshold:
                                    reused_tid = rid
                                    break
                            else:
                                # fallback to name-only reattach
                                reused_tid = rid
                                break
                    if reused_tid is not None:
                        # restore track
                        tid = reused_tid
                        tr = removed_tracks.pop(tid)
                        tr['bbox'] = (x, y, w, h)
                        tr['last_seen'] = current_ts
                        tr['skipped'] = 0
                        tr['vel'] = (0.0, 0.0)
                        tr['last_conf'] = confidence
                        tr['votes'] = deque(maxlen=history_len)
                        if candidate_name != "Unknown":
                            tr['votes'].append(candidate_name)
                        # compute brief appearance descriptor to help later re-attach
                        try:
                            color_roi = img[y:y+h, x:x+w]
                            tr['hist'] = compute_color_hist(color_roi)
                        except Exception:
                            tr['hist'] = None
                        tracks[tid] = tr
                        matched_track_ids.add(tid)
                    else:
                        # If detection is Unknown, try to assign it to an existing locked track
                        reassigned = False
                        if candidate_name == "Unknown":
                            for otid, otr in tracks.items():
                                if not otr.get('locked'):
                                    continue
                                # compare against expanded locked bbox or hist
                                comp_bbox = expand_bbox(otr.get('bbox', (0,0,0,0)), factor=1.6)
                                if iou(comp_bbox, (x, y, w, h)) >= 0.08:
                                    # accept as belonging to this locked track
                                    tr = otr
                                    tr['last_seen'] = current_ts
                                    tr['last_conf'] = confidence
                                    # do not append Unknown to votes
                                    # refresh grace when small detection seen
                                    tr['grace_left'] = locked_grace_frames
                                    tracks[otid] = tr
                                    matched_track_ids.add(otid)
                                    reassigned = True
                                    break
                                # fallback: histogram similarity
                                try:
                                    now_hist_tmp = now_hist
                                    if otr.get('hist') is not None and now_hist_tmp is not None:
                                        dist_tmp = cv2.compareHist(otr['hist'], now_hist_tmp, cv2.HISTCMP_BHATTACHARYYA)
                                    else:
                                        dist_tmp = 1.0
                                except Exception:
                                    dist_tmp = 1.0
                                if dist_tmp <= hist_threshold:
                                    tr = otr
                                    tr['last_seen'] = current_ts
                                    tr['last_conf'] = confidence
                                    # do not append Unknown to votes
                                    tr['grace_left'] = locked_grace_frames
                                    tracks[otid] = tr
                                    matched_track_ids.add(otid)
                                    reassigned = True
                                    break
                        if reassigned:
                            continue
                        # create new track (but avoid new tracks for small Unknown detections)
                        if candidate_name == "Unknown" and (w * h) < min_area_for_unknown_track:
                            # ignore small/low-quality unknown detections to reduce noise
                            continue
                        # otherwise create new track
                        tid = next_track_id
                        next_track_id += 1
                        tr = {
                            'votes': deque(maxlen=history_len),
                            'confirmed_name': None,
                            'last_top': None,
                            'candidate_count': 0,
                            'bbox': (x, y, w, h),
                            'skipped': 0,
                            'last_seen': current_ts,
                            'last_conf': confidence
                        }
                        # initialize a simple Kalman filter for center tracking
                        try:
                            kf = cv2.KalmanFilter(4, 2)
                            kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
                            kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
                            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
                            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
                            kf.errorCovPost = np.eye(4, dtype=np.float32)
                            cx = int(x + w/2)
                            cy = int(y + h/2)
                            kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0.0], [0.0]], np.float32)
                            tr['kf'] = kf
                        except Exception:
                            tr['kf'] = None
                        tr['votes'].append(candidate_name)
                        tracks[tid] = tr
                        matched_track_ids.add(tid)

            # Predict for unmatched tracks and draw all tracks; cleanup stale ones
            to_delete = []
            for tid, tr in list(tracks.items()):
                if tid not in matched_track_ids:
                    # no detection matched this track this frame
                    # For locked tracks, use a short grace period where we keep the bbox
                    # stable and do not increment the skipped counter. This prevents the
                    # tracker from removing or drifting the box when the person briefly
                    # looks away or turns.
                    if tr.get('locked'):
                        grace = tr.get('grace_left', 0)
                        if grace > 0:
                            tr['grace_left'] = grace - 1
                            # keep same bbox and do not increase skipped
                            ox, oy, ow, oh = tr.get('bbox', (0, 0, 0, 0))
                            tr['bbox'] = (ox, oy, ow, oh)
                        else:
                            # grace exhausted: proceed to prediction but less aggressively
                            tr['skipped'] = tr.get('skipped', 0) + 1
                            ox, oy, ow, oh = tr.get('bbox', (0, 0, 0, 0))
                            if tr.get('kf') is not None:
                                try:
                                    pred = tr['kf'].predict()
                                    pcx, pcy = int(pred[0][0]), int(pred[1][0])
                                    px = int(pcx - ow/2)
                                    py = int(pcy - oh/2)
                                    tr['bbox'] = (px, py, ow, oh)
                                except Exception:
                                    tr['bbox'] = (ox, oy, ow, oh)
                            else:
                                tr['bbox'] = (ox, oy, ow, oh)
                    else:
                        # not locked: normal behavior
                        tr['skipped'] = tr.get('skipped', 0) + 1
                        ox, oy, ow, oh = tr.get('bbox', (0, 0, 0, 0))
                        if tr.get('kf') is not None:
                            try:
                                pred = tr['kf'].predict()
                                pcx, pcy = int(pred[0][0]), int(pred[1][0])
                                px = int(pcx - ow/2)
                                py = int(pcy - oh/2)
                                tr['bbox'] = (px, py, ow, oh)
                            except Exception:
                                tr['bbox'] = (ox, oy, ow, oh)
                        else:
                            tr['bbox'] = (ox, oy, ow, oh)
                        
                # remove track if unseen for too long or predicted too many frames
                effective_forget = forget_timeout_locked if tr.get('locked') else forget_timeout
                if (current_ts - tr.get('last_seen', 0)) > effective_forget or tr.get('skipped', 0) > max_predict:
                    # move to removed_tracks for potential re-attachment
                    removed_tracks[tid] = tr
                    to_delete.append(tid)
                    continue
                display_name = tr.get('confirmed_name') or "Unknown"
                draw_x, draw_y, draw_w, draw_h = tr.get('bbox', (0, 0, 0, 0))
                # Always use green for known, red for unknown (no yellow lock indicator)
                color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
                thickness = 2
                cv2.rectangle(img, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), color, thickness)
                cv2.putText(img, display_name, (draw_x+5, draw_y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                last_conf = tr.get('last_conf')
                if last_conf is not None:
                    conf_text = f"Conf: {last_conf:.1f}"
                    cv2.putText(img, conf_text, (draw_x+5, draw_y+draw_h-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            for tid in to_delete:
                tracks.pop(tid, None)
            
            # Toggle frame processing flag
            process_this_frame = not process_this_frame
            
            # Display performance metrics
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Removed on-screen "Skipped" counter per user request
            
            # Show the frame and handle simple key commands
            cv2.imshow('Face Recognition', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('l'):
                # Lock nearest track to center (manual override)
                if tracks:
                    fh, fw = img.shape[:2]
                    cx_frame, cy_frame = fw // 2, fh // 2
                    best_tid = None
                    best_dist = float('inf')
                    for tid, tr in tracks.items():
                        bx, by, bw, bh = tr.get('bbox', (0,0,0,0))
                        tcx = bx + bw/2
                        tcy = by + bh/2
                        d = (tcx - cx_frame)**2 + (tcy - cy_frame)**2
                        if d < best_dist:
                            best_dist = d
                            best_tid = tid
                    if best_tid is not None:
                        tr = tracks[best_tid]
                        tr['locked'] = True
                        tr['grace_left'] = locked_grace_frames
                        tr['lock_time'] = time.time()
                        tracks[best_tid] = tr
            elif key == ord('u'):
                # Unlock nearest track (or all if shift+u pressed) - simple toggle
                if tracks:
                    fh, fw = img.shape[:2]
                    cx_frame, cy_frame = fw // 2, fh // 2
                    best_tid = None
                    best_dist = float('inf')
                    for tid, tr in tracks.items():
                        bx, by, bw, bh = tr.get('bbox', (0,0,0,0))
                        tcx = bx + bw/2
                        tcy = by + bh/2
                        d = (tcx - cx_frame)**2 + (tcy - cy_frame)**2
                        if d < best_dist:
                            best_dist = d
                            best_tid = tid
                    if best_tid is not None:
                        tr = tracks[best_tid]
                        tr['locked'] = False
                        tr['grace_left'] = 0
                        tracks[best_tid] = tr
        
        logger.info("Face recognition stopped")

        # Cleanup
        if 'cam' in locals():
            try:
                cam.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


        