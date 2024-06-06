from flask import Flask, render_template, request
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)


def pad_truncate_sequence(seq, max_len):
    if seq.shape[1] > max_len:
        return seq[:, :max_len]
    else:
        return np.pad(seq, ((0, 0), (0, max_len - seq.shape[1])), 'constant')


# Ses dosyasını kaydet ve modeli kullanarak tahmin yap
def predict_audio(file_path, model):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        fixed_length = 173
        mfccs = pad_truncate_sequence(mfccs, fixed_length)
        mfccs = mfccs.flatten()
        mfccs = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(mfccs)
        predicted_label = np.argmax(prediction)
        label_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad'}
        return label_dict[predicted_label]
    except Exception as e:
        return str(e)

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Ses kaydetme ve tahmin yapma endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        duration = 5  # 5 saniye kayıt
        sample_rate = 22050  # Örnek hızı
        file_path = 'recorded_audio.wav'
        try:
            # Ses kaydı yap
            print("Ses kaydediliyor. Lütfen konuşun...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
            sd.wait()  # Kaydetme tamamlanana kadar bekle
            # Ses dosyasını kaydet
            sf.write(file_path, audio, sample_rate)
            # Modeli yükle
            loaded_model = load_model('audio_classification_model.h5')
            # Tahmin yap
            prediction = predict_audio(file_path, loaded_model)
            return prediction
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
