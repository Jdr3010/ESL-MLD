from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import threading
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from googletrans import Translator
import mediapipe as mp
import time

# ✅ Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ✅ Load models
models = {
    "ASL": load_model(r"C:\Users\John Daniel\asl_lstm.keras"),
    "BSL": load_model(r"C:\Users\John Daniel\bsl_sign_language_model1.keras"),
    "ISL": load_model(r"C:\Users\John Daniel\indsign_language_lstm_model.keras"),
    "LSFB": load_model(r"C:\Users\John Daniel\Documents\fsl\fsl_lstm_model.keras"),
    "LSE": load_model(r"C:\Users\John Daniel\spanishsl_model.keras")
}

# ✅ Model shapes
model_shapes = {
    "ASL": (100, 63),
    "BSL": (369, 207),
    "ISL": (192, 207),
    "LSFB": (100, 1659),
    "LSE": (124, 207)
}

# ✅ Load labels
csv_files = {
    "ASL": r"C:\Users\John Daniel\Downloads\ASL\how2sign_realigned_train.csv",
    "BSL": r"C:\Users\John Daniel\Downloads\BSL\bsl_sign_to_word.xlsx",
    "ISL": r"C:\Users\John Daniel\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL Corpus sign glosses.csv",
    "LSFB": r"C:\Users\John Daniel\Documents\fsl\instances.csv",
    "LSE": r"C:\Users\John Daniel\Downloads\lse\videos_ref_annotations.csv"
}

def load_labels():
    """Load CSV and Excel label files into memory."""
    labels = {}
    for lang, path in csv_files.items():
        try:
            if lang == "ASL":
                df = pd.read_csv(path, delimiter="\t", encoding="utf-8")
                df.columns = ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE']
                labels[lang] = df["SENTENCE"].tolist()
            elif lang == "BSL":
                df = pd.read_excel(path, engine='openpyxl')
                labels[lang] = df["WORD"].tolist()
            elif lang == "ISL":
                df = pd.read_csv(path, encoding="utf-8")
                labels[lang] = df["SIGN GLOSSES"].tolist()
            elif lang == "LSFB":
                df = pd.read_csv(path, encoding="utf-8")
                labels[lang] = df["sign"].tolist()
            elif lang == "LSE":
                df = pd.read_csv(path, encoding="utf-8")
                labels[lang] = df["LABEL"].tolist()

            print(f"✅ Loaded labels for {lang}: {len(labels[lang])} items")

        except Exception as e:
            print(f"❌ Error loading labels for {lang}: {e}")
    return labels

labels = load_labels()

# ✅ Translator
translator = Translator()

# ✅ Global variables
detected_word = ""
translated_word = ""
active_language = "ISL"  # Default language
target_language = "en"   # Default translation language

# ✅ Threading variables
language_lock = threading.Lock()
stream_lock = threading.Lock()
switching_event = threading.Event()

# ✅ Serve the index.html
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the frontend HTML."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ✅ Switch language dynamically
@app.post("/switch_language/")
async def switch_language(request: Request):
    """Switch the active language immediately and trigger 2s delay."""
    global active_language, switching_event, detected_word, translated_word

    data = await request.json()
    new_lang = data.get("language")

    if new_lang not in models:
        raise HTTPException(status_code=400, detail="Invalid language.")

    with language_lock:
        # 🔥 Switch language and clear detected words
        active_language = new_lang
        detected_word = ""
        translated_word = ""

        # 🔥 Trigger switching event with a 2s delay
        switching_event.set()
        print(f"✅ Switching to {new_lang}. Waiting for 2s...")
        time.sleep(2)
        switching_event.clear()

    return JSONResponse(content={"message": f"Switched to {new_lang}"})

# ✅ Get the detected and translated word
@app.get("/get_detected_word/")
async def get_detected_word():
    """Get the current detected and translated word."""
    with language_lock:
        return JSONResponse(content={
            "detected_word": detected_word,
            "translated_word": translated_word
        })

# ✅ Translate the detected word
@app.get("/translate/")
async def translate_word(target_lang: str = Query(...)):
    """Translate the detected word to the target language."""
    global target_language, translated_word

    target_language = target_lang

    if detected_word:
        try:
            translation = await run_in_threadpool(lambda: translator.translate(detected_word, dest=target_lang))
            translated_word = translation.text
            print(f"✅ Translated to {target_lang}: {translated_word}")
        except Exception as e:
            print(f"❌ Translation error: {e}")
            translated_word = "Translation failed"

    return JSONResponse(content={"translated_word": translated_word})

# ✅ Extract keypoints
def extract_keypoints(frame, lang):
    """Extract face, hand, and pose keypoints."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(min_detection_confidence=0.5)

    results_hands = hands_model.process(frame_rgb)

    hand_kps = np.zeros((42, 3))

    if results_hands.multi_hand_landmarks:
        for i, hand in enumerate(results_hands.multi_hand_landmarks):
            kps = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            if i == 0:
                hand_kps[:21] = kps
            else:
                hand_kps[21:] = kps

    combined = hand_kps.flatten()

    input_shape = model_shapes.get(lang, (100, 63))
    
    if combined.shape[0] < np.prod(input_shape):
        combined = np.pad(combined, (0, np.prod(input_shape) - combined.shape[0]), mode='constant')

    return np.reshape(combined, input_shape)

# ✅ Real-time video stream with proper model switching
def video_stream():
    global detected_word, translated_word

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Skip detection during switching event
        if switching_event.is_set():
            time.sleep(2)
            continue

        keypoints = extract_keypoints(frame, active_language)
        batch_data = np.expand_dims(keypoints, axis=0)

        predictions = models[active_language].predict(batch_data, verbose=0)
        detected_word = labels[active_language][np.argmax(predictions)]

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

@app.get("/video_feed/")
async def video_feed():
    """Stream video with detection and translation."""
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# ✅ Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9000)
