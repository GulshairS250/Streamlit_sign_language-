"""
Streamlit realtime sign-language detection using the trained LSTM (.h5) and
MediaPipe Holistic keypoints — same pipeline as RealTimeSignLanguageDetection.ipynb.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "model.h5"
DEFAULT_WEIGHTS = BASE_DIR / "model_weights.h5"
SEQUENCE_LENGTH = 30
FEATURE_LEN = 1662
DEFAULT_ACTIONS = ["cat", "food", "help"]


def extract_keypoints(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()
    else:
        pose = np.zeros(33 * 4)

    if results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        ).flatten()
    else:
        face = np.zeros(468 * 3)

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


def draw_styled_landmarks(image: np.ndarray, results, mp_holistic, mp_drawing) -> None:
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def draw_prob_bars(
    frame: np.ndarray, probs: np.ndarray, labels: list[str], bar_color=(33, 150, 243)
) -> None:
    bar_h = 22
    spacing = 32
    for i, p in enumerate(probs):
        y0 = 12 + i * spacing
        cv2.rectangle(frame, (8, y0), (8 + int(p * 280), y0 + bar_h), bar_color, -1)
        name = labels[i] if i < len(labels) else f"class_{i}"
        cv2.putText(
            frame,
            f"{name}: {p:.2f}",
            (12, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


@st.cache_resource
def load_sign_model(model_path_str: str, weights_path_str: str) -> tf.keras.Model:
    model_path = Path(model_path_str)
    weights_path = Path(weights_path_str)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    if weights_path.is_file():
        model.load_weights(str(weights_path))
    return model


def resolve_labels(user_text: str, num_classes: int) -> list[str]:
    parts = [p.strip() for p in user_text.split(",") if p.strip()]
    if len(parts) == num_classes:
        return parts
    return [f"class_{i}" for i in range(num_classes)]


def cleanup_camera_session() -> None:
    st.session_state.run = False
    h = st.session_state.get("holistic")
    if h is not None:
        try:
            h.close()
        except Exception:
            pass
        st.session_state.holistic = None
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.cap = None
    st.session_state.sequence = []


def main() -> None:
    st.set_page_config(page_title="Sign language (LSTM)", layout="centered")
    st.title("Realtime sign language detection")
    st.caption(
        "Webcam + MediaPipe Holistic + your `model.h5` / `model_weights.h5` "
        "(same setup as RealTimeSignLanguageDetection.ipynb)."
    )

    if "run" not in st.session_state:
        st.session_state.run = False
    if "sequence" not in st.session_state:
        st.session_state.sequence = []
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "holistic" not in st.session_state:
        st.session_state.holistic = None

    with st.sidebar:
        st.header("Model paths")
        model_path = st.text_input("model.h5 path", value=str(DEFAULT_MODEL))
        weights_path = st.text_input("model_weights.h5 path", value=str(DEFAULT_WEIGHTS))
        st.header("Class names")
        default_csv = ",".join(DEFAULT_ACTIONS)
        label_text = st.text_input(
            "Labels (comma-separated, must match output size)",
            value=default_csv,
            help="Notebook default: cat, food, help",
        )
        cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
        frame_delay = st.slider("Frame delay (seconds)", 0.02, 0.12, 0.04, 0.01)

    try:
        model = load_sign_model(model_path, weights_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    num_out = int(model.output_shape[-1])
    labels = resolve_labels(label_text, num_out)
    if len([p for p in label_text.split(",") if p.strip()]) != num_out:
        st.warning(
            f"Model has {num_out} outputs; using generic names "
            f"({labels[0]}, …) until you enter {num_out} comma-separated labels."
        )

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start webcam", type="primary"):
            cleanup_camera_session()
            st.session_state.run = True
            st.session_state.sequence = []
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(f"Could not open camera {cam_index}.")
                st.session_state.run = False
            else:
                st.session_state.cap = cap
                st.session_state.holistic = mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            st.rerun()
    with c2:
        if st.button("Stop"):
            cleanup_camera_session()
            st.rerun()

    frame_slot = st.empty()
    status_slot = st.empty()

    if st.session_state.run and st.session_state.cap and st.session_state.holistic:
        cap = st.session_state.cap
        holistic = st.session_state.holistic
        ret, frame = cap.read()
        if not ret:
            status_slot.error("Failed to read from webcam.")
            cleanup_camera_session()
            st.stop()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True
        out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        draw_styled_landmarks(out_bgr, results, mp_holistic, mp_drawing)

        kp = extract_keypoints(results)
        if kp.shape[0] != FEATURE_LEN:
            status_slot.error(f"Keypoint size {kp.shape[0]} != expected {FEATURE_LEN}.")
        else:
            st.session_state.sequence.append(kp)
            st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LENGTH:]
            if len(st.session_state.sequence) == SEQUENCE_LENGTH:
                batch = np.expand_dims(np.array(st.session_state.sequence), axis=0)
                probs = model.predict(batch, verbose=0)[0]
                draw_prob_bars(out_bgr, probs, labels)
                best_i = int(np.argmax(probs))
                best_l = labels[best_i] if best_i < len(labels) else f"class_{best_i}"
                status_slot.markdown(f"**Prediction:** `{best_l}`  \n**Confidence:** {probs[best_i]:.2f}")
            else:
                need = SEQUENCE_LENGTH - len(st.session_state.sequence)
                status_slot.info(f"Collecting frames… **{need}** more until first prediction.")

        frame_slot.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(float(frame_delay))
        st.rerun()
    else:
        st.info("Click **Start webcam** and allow the browser to use your camera (if prompted).")
        st.markdown(
            "```bash\npip install -r requirements.txt\nstreamlit run app.py\n```"
        )


if __name__ == "__main__":
    main()
