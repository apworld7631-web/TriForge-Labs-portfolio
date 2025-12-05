import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from flask import Flask, jsonify
from flask_cors import CORS

"""
facial_server.py
----------------
Webcam + MediaPipe FaceMesh demo that estimates a simple facial
stress/fatigue score and exposes it via:

    GET http://127.0.0.1:5000/facial_score  ->  {"facial_score": <0-100>}

The frontend calls this API every few seconds and fuses it with
interaction-based cognitive load.
"""

# ---------- Global state ----------
facial_score = 0        # 0–100
running = True

# ---------- MediaPipe FaceMesh setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks indices for approximate left eye EAR
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133


def eye_aspect_ratio(landmarks, image_w, image_h):
    """Compute an eye-aspect-ratio (EAR)-like metric for the left eye."""
    top = landmarks[LEFT_EYE_TOP]
    bottom = landmarks[LEFT_EYE_BOTTOM]
    left = landmarks[LEFT_EYE_LEFT]
    right = landmarks[LEFT_EYE_RIGHT]

    top = np.array([top.x * image_w, top.y * image_h])
    bottom = np.array([bottom.x * image_w, bottom.y * image_h])
    left = np.array([left.x * image_w, left.y * image_h])
    right = np.array([right.x * image_w, right.y * image_h])

    vert = np.linalg.norm(top - bottom)
    horiz = np.linalg.norm(left - right)
    if horiz == 0:
        return 0.0
    return vert / horiz   # smaller → more closed eye


def heuristic_facial_score(ear_history):
    """
    Stand-in for a CNN-based stress model:

    - Compute average EAR over a sliding window.
    - Map to 0–100 fatigue score.
    - Lower EAR (eyes more closed) → higher fatigue.
    """
    if not ear_history:
        return 0

    avg_ear = sum(ear_history) / len(ear_history)

    # Approximate typical EAR range for open vs more closed eyes
    min_ear = 0.18
    max_ear = 0.32

    norm = (avg_ear - min_ear) / (max_ear - min_ear + 1e-6)
    norm = max(0.0, min(1.0, norm))     # clamp 0–1
    fatigue_score = (1.0 - norm) * 100  # invert
    return int(fatigue_score)


def camera_loop():
    """Continuously read webcam, run FaceMesh, update global facial_score."""
    global facial_score, running

    cap = cv2.VideoCapture(0)
    ear_values = []
    timestamps = []
    window_seconds = 5

    print("[Camera] Started. Press 'q' in the camera window to stop.")

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_score = 0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            ear = eye_aspect_ratio(landmarks_list, w, h)
            t_now = time.time()

            ear_values.append(ear)
            timestamps.append(t_now)

            # keep only last window_seconds
            while timestamps and (t_now - timestamps[0] > window_seconds):
                timestamps.pop(0)
                ear_values.pop(0)

            current_score = heuristic_facial_score(ear_values)

            # sparse landmark visualization
            for lm in landmarks_list[::10]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
        else:
            current_score = 0  # no face detected

        facial_score = current_score

        cv2.putText(frame, f"Facial Stress/Fatigue Score: {facial_score}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame,
                    "MediaPipe FaceMesh (input to CNN in full system)",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        cv2.imshow("Facial Load Demo (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Camera] Stopped.")


# ---------- Flask API ----------
from flask import Flask
app = Flask(__name__)
CORS(app)  # allow browser JS calls from localhost


@app.route("/facial_score", methods=["GET"])
def get_facial_score():
    """Return latest facial_score as JSON."""
    return jsonify({"facial_score": int(facial_score)})


def start_server():
    print("[Server] Running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    # start camera loop in background thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    try:
        start_server()
    finally:
        running = False
        cam_thread.join()
        print("[Main] Shutdown complete.")
