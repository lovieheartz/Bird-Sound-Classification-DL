import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, redirect
from werkzeug.utils import secure_filename
import base64
from warnings import filterwarnings

filterwarnings('ignore')

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
with open('prediction.json', 'r') as f:
    prediction_dict = json.load(f)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_audio(file_path):
    audio, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc, axis=1)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=2)
    tensor = tf.convert_to_tensor(mfcc, dtype=tf.float32)

    pred = model.predict(tensor)
    label_index = np.argmax(pred)
    confidence = round(float(np.max(pred)) * 100, 2)
    class_name = prediction_dict[str(label_index)]

    return class_name, confidence

def encode_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 300))
    _, buffer = cv2.imencode('.jpg', img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = ""
    if request.method == 'POST':
        file = request.files.get('audio')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_class, confidence = predict_audio(filepath)
            image_path = os.path.join('Inference_Images', f'{predicted_class}.jpg')
            image_encoded = encode_image(image_path)

            result_html = f"""
                <div class="result">
                    <h2>{confidence:.2f}% Match Found</h2>
                    <img src="{image_encoded}" alt="{predicted_class}" />
                    <h1>{predicted_class}</h1>
                </div>
            """

    return Response(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bird Sound Classification</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Segoe UI', sans-serif;
                    background: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb') no-repeat center center fixed;
                    background-size: cover;
                    color: #fff;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: flex-start;
                    min-height: 100vh;
                }}
                .container {{
                    background-color: rgba(0, 0, 0, 0.7);
                    margin-top: 60px;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.5);
                    width: 90%;
                    max-width: 500px;
                    text-align: center;
                }}
                h1 {{
                    color: #00ffe1;
                    margin-bottom: 25px;
                    font-size: 2em;
                }}
                form {{
                    margin-top: 20px;
                }}
                input[type="file"] {{
                    padding: 10px;
                    background-color: #222;
                    color: #fff;
                    border-radius: 8px;
                    border: 1px solid #444;
                    width: 100%;
                    margin-bottom: 15px;
                }}
                input[type="submit"] {{
                    background-color: #00bcd4;
                    color: white;
                    padding: 12px 25px;
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: background 0.3s;
                }}
                input[type="submit"]:hover {{
                    background-color: #0097a7;
                }}
                .result {{
                    margin-top: 40px;
                }}
                .result img {{
                    margin-top: 20px;
                    border-radius: 15px;
                    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
                    max-width: 100%;
                    height: auto;
                }}
                .result h2 {{
                    color: #ff9800;
                    margin-bottom: 10px;
                }}
                .result h1 {{
                    color: #4caf50;
                    margin-top: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Bird Sound Classification</h1>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="audio" accept=".wav, .mp3" required><br>
                    <input type="submit" value="Classify">
                </form>
                {result_html}
            </div>
        </body>
        </html>
    """, mimetype='text/html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
