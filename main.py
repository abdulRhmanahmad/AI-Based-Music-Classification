from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import librosa
import numpy as np
from keras.models import load_model
from scipy import stats
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

model_path = "model_cnn3.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model_cnn3 = load_model(model_path)


def load_audio_file(file_path, target_sr=22050):
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in [".mp3", ".m4a"]:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio, sr
        elif ext == ".wav":
            audio, sample_rate = librosa.load(file_path, sr=target_sr)
            return audio, sample_rate
        else:
            raise ValueError(
                "Unsupported file format. Supported formats: WAV, MP3, M4A."
            )
    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {str(e)}")


def extract_features(
    file_path, fs=22050, n_mfcc=13, n_fft=2048, hop_length=512, segment_length=3
):
    audio, sample_rate = load_audio_file(file_path, target_sr=fs)
    segment_length_samples = int(fs * segment_length)
    num_segments = len(audio) // segment_length_samples
    num_segments = max(num_segments, 1)
    mfccs_per_segment = 130
    features = []
    for seg in range(num_segments):
        start_sample = seg * segment_length_samples
        end_sample = start_sample + segment_length_samples
        if end_sample <= len(audio):
            try:
                mfcc = librosa.feature.mfcc(
                    y=audio[start_sample:end_sample],
                    sr=sample_rate,
                    n_fft=n_fft,
                    hop_length=math.floor(segment_length_samples / mfccs_per_segment),
                    n_mfcc=n_mfcc,
                )
                mfcc = mfcc.T
                if len(mfcc) < mfccs_per_segment:
                    padding = mfccs_per_segment - len(mfcc)
                    mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode="constant")
                elif len(mfcc) > mfccs_per_segment:
                    mfcc = mfcc[:mfccs_per_segment, :]
                features.append(mfcc)
            except Exception as e:
                print(f"Error extracting features for segment {seg}: {str(e)}")
                continue
    features = np.array(features)
    num_rows, num_columns, num_channels = features.shape[1], features.shape[2], 1
    features = features.reshape(features.shape[0], num_rows, num_columns, num_channels)
    return features, sample_rate


genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


@app.post("/predict_genre/")
async def predict_genre(request: Request):
    try:
        data = await request.json()
        file_path = data.get("file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="File path is required")

        # تحويل الفواصل العكسية إلى فواصل أمامية
        file_path = file_path.replace("\\", "/")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        features, _ = extract_features(file_path)
        prediction = model_cnn3.predict(features)
        predicted_class = np.argmax(prediction, axis=1)
        mode_result = stats.mode(predicted_class, axis=None)

        if isinstance(mode_result.mode, np.ndarray):
            final_class_index = int(mode_result.mode[0])
        else:
            final_class_index = int(mode_result.mode)

        final_genre = genres[final_class_index]
        return JSONResponse(content={"predicted_genre": final_genre})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, workers=True)
