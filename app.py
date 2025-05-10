# app.py
import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, send_from_directory, jsonify
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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
    filename = ""
    if request.method == 'POST':
        # Check if request is XHR (AJAX)
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
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
                    <h3><i class="fas fa-volume-up"></i> Uploaded Audio</h3>
                    <audio controls>
                        <source src="/uploads/{filename}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <h2> {confidence:.2f}% Match</h2>
                    <img src="{image_encoded}" alt="{predicted_class}" />
                    <h1>{predicted_class}</h1>
                </div>
            """
            
            # Return only the result HTML if it's an AJAX request
            if is_ajax or request.headers.get('Accept') == 'application/json':
                return jsonify({"result_html": result_html})

    return Response(r"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TWEETIFY</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
      * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
      }

      body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
          background-color: black;
          color: #e7e7e7;
          min-height: 100vh;
          line-height: 1.5;
      }

      /* GRADIENT EFFECT */
      .image-gradient {
          position: absolute;
          top: 0;
          right: 0;
          z-index: -1;
      }

      .layer-blur {
          height: 0;
          width: 30rem;
          position: absolute;
          top: 20%;
          right: 0;
          box-shadow: 0 0 700px 15px white;
          rotate: -30deg;
          z-index: -1;
      }
      /* CONTAINER */
      .container {
          width: 100%;
          margin: 0 auto;
          padding: 0 2rem;
          position: relative;
          overflow: hidden;
      }

      /* HEADER */
      header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem 5rem;
          z-index: 999;
      }

      header h1 {
          margin: 0;
          font-size: 3rem;
          font-weight: 300;
      }

      nav {
          display: flex;
          align-items: center;
          gap: 3rem;
          margin-left: -5%;
      }

      nav a {
          font-size: 1rem;
          letter-spacing: 0.1rem;
          transition: color 0.2s ease;
          text-decoration: none;
          color: inherit;
      }

      nav a:hover {
          color: #a7a7a7;
      }

      .btn-signin {
          background-color: #a7a7a7;
          color: black;
          padding: 0.8rem 2rem;
          border-radius: 50px;
          border: none;
          font-size: 1rem;
          font-weight: 500;
          transition: background-color 0.2s ease;
          cursor: pointer;
      }

      .btn-signin:hover {
          background-color: white;
      }

      /* MAIN CONTENT */
      main {
          display: flex;
          flex-direction: column;
          justify-content: center;
          min-height: calc(90vh - 6rem);
      }

      .content {
          max-width: 40rem;
          margin-left: 10%;
          margin-bottom: 3rem;
          z-index: 999;
      }

      .tag-box{
          position: relative;
          width: 18rem;
          height: 2.5rem;
          border-radius: 50px;
          background: linear-gradient(to right, #656565, #7f42a7, #6600c5, #5300a0, #757575, #656565);
          background-size: 200%;
          animation: gradientAnimation 2.5s linear infinite;
          box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
      }

      @keyframes gradientAnimation {
          0% {
              background-position: 0%;
          }
          50% {
              background-position: 100%;
          }
          100% {
              background-position: 0%;
          }
      }

      .tag-box .tag{
          position: absolute;
          inset: 3px 3px 3px 3px;
          background-color: black;
          border-radius: 50px;
          display: flex;
          justify-content: center;
          align-items: center;
          transition: 0.5s ease;
          cursor: pointer;
      }

      .tag-box .tag:hover{
          color: #b38ed4;
      }

      .content h1 {
          font-size: 4rem;
          font-weight: 600;
          letter-spacing: 0.1em;
          margin: 2rem 0;
          line-height: 1.2;
          text-shadow: 0 0 10px rgba(128,128,128,0.418);
      }

      .description {
          font-size: 1.2rem;
          letter-spacing: 0.05em;
          max-width: 35rem;
          color: gray;
      }

      .buttons {
          display: flex;
          gap: 1rem;
          margin-top: 3rem;
      }

      .btn-get-started {
          text-decoration: none;
          border: 1px solid #2a2a2a;
          padding: 0.7rem 1.2rem;
          border-radius: 50px;
          font-size: 1.2rem;
          font-weight: 600;
          letter-spacing: 0.1em;
          transition: background-color 0.2s ease;
          color: #e7e7e7;
      }

      .btn-get-started:hover {
          background-color: #1a1a1a;
      }

      .btn-signin-main {
          text-decoration: none;
          background-color: lightgray;
          color: black;
          padding: 0.6rem 2.5rem;
          border-radius: 50px;
          font-size: 1.2rem;
          font-weight: 600;
          letter-spacing: 0.1em;
          transition: background-color 0.2s ease;
          cursor: pointer;
          border: none;
      }

      .btn-signin-main:hover {
          background-color: gray;
      }

      .robot-3d {
          position: absolute;
          top: 8%;
          right: -40%;
          z-index: -1;
      }

      /* File upload styling */
      .container-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        padding: 20px 0;
      }
      
      .classify-section {
          display: none;
          opacity: 0;
          transform: translateY(20px);
          transition: opacity 0.5s ease, transform 0.5s ease;
          width: 100%;
      }

      .classify-section.active {
          display: flex;
          justify-content: center;
          align-items: center;
          opacity: 1;
          transform: translateY(0);
      }

      .upload-container {
          background-color: rgba(30, 41, 59, 0.8);
          padding: 30px;
          border-radius: 15px;
          box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 100%;
          max-width: 900px;
      }

      .file-input-wrapper {
          position: relative;
          width: 100%;
          margin-bottom: 20px;
      }

      .file-input {
          padding: 12px;
          border-radius: 50px;
          background-color: #334155;
          color: #e7e7e7;
          border: 1px solid #444;
          width: 100%;
          font-size: 1rem;
          cursor: pointer;
      }

      /* Updated Button Styles */
      .btn-container {
          display: flex;
          justify-content: center;
          gap: 20px;
          width: 100%;
      }

      .submit-btn, .reset-btn {
          background: linear-gradient(45deg, #7f42a7, #6600c5);
          color: white;
          padding: 12px 25px;
          border: none;
          border-radius: 50px;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.3s ease;
          width: 100%;
          max-width: 200px;
          position: relative;
          overflow: hidden;
          box-shadow: 0 4px 15px rgba(102, 0, 197, 0.3);
          cursor: pointer;
          z-index: 999;

      }


      .submit-btn:hover , .reset-btn:hover{
        opacity: 0.8;
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(102, 0, 197, 0.4);
      }

      /* Button animation */
      .submit-btn::before, .reset-btn::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
          transition: 0.5s;
      }

      .submit-btn:hover::before, .reset-btn:hover::before {
          left: 100%;
      }

      /* Reset button specific color */
      .reset-btn {
          background: linear-gradient(45deg, #8e44ad, #9b59b6);
      }

      #loading {
          display: none;
          margin-top: 20px;
          color: #b38ed4;
      }

      .result {
          margin-top: 30px;
          text-align: center;
          width: 100%;
          background-color: rgba(51, 65, 85, 0.6);
          padding: 20px;
          border-radius: 15px;
          animation: fadeIn 1s ease-in-out;
      }

      @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
      }

      .result h2 {
          color: #b38ed4;
          margin-bottom: 10px;
          font-size: 1.5rem;
      }

      .result h3 {
          color: #d2b8e3;
          margin-top: 15px;
          font-size: 1.5rem;
      }

      .result h1 {
          color: #f1f5f9;
          margin-top: 10px;
          font-size: 2rem;
      }

      .result img {
          margin-top: 20px;
          border-radius: 15px;
          box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
          max-width: 100%;
          height: auto;
      }

      /* Loading animation */
      @keyframes pulse {
          0% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.1); opacity: 0.7; }
          100% { transform: scale(1); opacity: 1; }
      }

      #loading .fa-spinner {
          animation: pulse 1.5s infinite;
          color: #b38ed4;
      }

      /* TABLET RESPONSIVE */
      @media(max-width: 1300px){
          header {
              padding: 1rem 0.5rem;
          }

          .content {
              margin-top: 10%;
          }

          .robot-3d {
              scale: 0.8;
              right: -15%;
          }
      }

      /* Footer Styles */
      footer {
          background-color: rgba(0, 0, 0, 0.7);
          padding: 3rem 0 1rem;
          margin-top: 4rem;
          position: relative;
          z-index: 10;
      }

      .footer-container {
          display: flex;
          flex-wrap: wrap;
          justify-content: space-between;
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 2rem;
      }

      .footer-section {
          flex: 1;
          min-width: 200px;
          margin-bottom: 2rem;
      }

      .footer-section.brand {
          flex: 2;
      }

      .footer-section h3 {
          font-size: 2rem;
          font-weight: 300;
          margin-bottom: 1rem;
          color: #e7e7e7;
      }

      .footer-section h4 {
          font-size: 1.2rem;
          margin-bottom: 1rem;
          color: #a7a7a7;
          font-weight: 500;
      }

      .footer-section p {
          color: #888;
          line-height: 1.6;
      }

      .footer-section ul {
          list-style: none;
      }

      .footer-section ul li {
          margin-bottom: 0.5rem;
      }

      .footer-section ul li a {
          color: #888;
          text-decoration: none;
          transition: color 0.2s ease;
      }

      .footer-section ul li a:hover {
          color: #e7e7e7;
      }

      .social-icons {
          display: flex;
          gap: 1rem;
          margin-bottom: 1.5rem;
      }

      .social-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background-color: #222;
          color: #a7a7a7;
          transition: all 0.3s ease;
      }

      .social-icon:hover {
          background-color: #7f42a7;
          color: white;
          transform: translateY(-3px);
      }

      .newsletter-form {
          display: flex;
          margin-top: 1rem;
      }

      .newsletter-form input {
          flex: 1;
          padding: 0.8rem;
          border: none;
          border-radius: 4px 0 0 4px;
          background-color: #222;
          color: #e7e7e7;
      }

      .newsletter-form button {
          padding: 0.8rem 1.2rem;
          border: none;
          border-radius: 0 4px 4px 0;
          background: linear-gradient(to right, #7f42a7, #6600c5);
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
      }

      .newsletter-form button:hover {
          background: linear-gradient(to right, #6600c5, #5300a0);
      }

      .footer-bottom {
          text-align: center;
          padding-top: 2rem;
          margin-top: 1rem;
          border-top: 1px solid #333;
      }

      .footer-bottom p {
          color: #666;
          font-size: 0.9rem;
          margin-bottom: 0.5rem;
      }

      .footer-bottom a {
          color: #888;
          text-decoration: none;
          transition: color 0.2s ease;
      }

      .footer-bottom a:hover {
          color: #e7e7e7;
      }

      /* MOBILE RESPONSIVE */
      @media(max-width: 768px) {
          header {
              padding: 1rem 0.1rem;
          }

          nav {
              display: none;
          }

          header h1 {
              font-size: 2rem;
          }

          .btn-signin {
              padding: 0.6rem 1.5rem;
          }

          .content {
              margin-top: 5rem;
              margin-left: 5%;
          }

          .robot-3d {
              scale: 0.5;
              top: -30%;
              right: 28%;
          }

          .content {
              max-width: 30rem;
          }

          .tag-box {
              width: 12rem;
          }

          .content h1{
              font-size: 2.5rem;
          }

          .description {
              font-size: 1rem;
          }
          
          .btn-get-started {
              font-size: 0.8rem;
              padding: 0.5rem 1.2rem;
          }
          
          .btn-signin-main {
              font-size: 0.8rem;
              padding: 0.5rem 2rem;
          }

          .buttons {
              flex-direction: column;
              align-items: flex-start;
          }
          
          .footer-container {
              flex-direction: column;
              padding: 0 1rem;
          }
          
          .footer-section {
              margin-bottom: 1.5rem;
          }
          
          .newsletter-form {
              flex-direction: column;
          }
          
          .newsletter-form input {
              border-radius: 4px;
              margin-bottom: 0.5rem;
          }
          
          .newsletter-form button {
              border-radius: 4px;
          }

          .upload-container {
              padding: 20px;
          }
          
          /* Responsive button container for mobile */
          .btn-container {
              flex-direction: column;
              gap: 10px;
          }
      }
    </style>
  </head>
  <body>
   

    <div class="layer-blur"></div>

    <div class="container">
      <header>
        <h1 class="logo">TWEETIFY</h1>

        <nav>
          <a href="#">HOME</a>
          <a href="#">ABOUT</a>
          <a href="#resource-section">RESOURCES</a>
          <a href="#">CONTACT</a>
        </nav>
      </header>
      <main>
        <div class="content">
          <div class="tag-box">
            <div class="tag">INTRODUCING TWEETIFY</div>
          </div>
          <h1>
            DECODE <br />
            THE BIRD
          </h1>

          <p class="description">
            The smartest way to identify birds. <br> Upload bird sounds and classify
            species instantly using TWEETIFY — AI-powered bird sound classification.
          </p>

          <div class="buttons">
            <a href="#classify-section" class="btn-signin-main" onclick="showClassifySection()">Get Started &gt;</a>
          </div>
        </div>

        <!-- Classification section -->
        <div class="container-wrap">
          <div id="classify-section" class="classify-section">
            <div class="upload-container">
              <h2 style="margin-bottom: 20px; color: #b38ed4;">Upload Bird Sound</h2>
              <form id="uploadForm" method="POST" enctype="multipart/form-data">
                <div class="file-input-wrapper">
                  <input type="file" name="audio" accept=".wav, .mp3" required class="file-input">
                </div>
                <div class="btn-container">
                  <input type="submit" value="Classify" class="submit-btn">
                  <button type="button" class="reset-btn" onclick="resetForm()">Reset</button>
                </div>
              </form>
              
              <div id="loading">
                <p><i class="fas fa-spinner fa-spin"></i> Processing... Please wait</p>
              </div>
              
              
              <div id="resultContainer"></div>
            </div>
          </div>
        </div>
      </main>

      <spline-viewer
        class="robot-3d"
        url="https://prod.spline.design/AFY-KdQPdxzUUeA8/scene.splinecode"
      ></spline-viewer>
    </div>

    <footer>
      <div class="footer-container">
        <div class="footer-section brand">
          <h3>TWEETIFY</h3>
          <p>AI-powered bird sound classification</p>
        </div>
        <div class="footer-section links">
          <h4>Quick Links</h4>
          <ul>
            <li><a href="#">Home</a></li>
            <li><a href="#">About</a></li>
            <li><a href="#">Contact</a></li>
          </ul>
        </div>
        <div class="footer-section resources" id="resource-section">
          <h4>Resources</h4>
          <ul>
            <li><a href="https://github.com/lovieheartz/Bird-Sound-Classification-DL">Documentation</a></li>
            <li><a href="#">API</a></li>
            <li><a href="https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022">Bird Database</a></li>
            <li><a href="#">Research</a></li>
          </ul>
        </div>
        <div class="footer-section connect">
          <h4>Connect</h4>
          <div class="social-icons">
            <a href="#" class="social-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path></svg>
            </a>
            <a href="#" class="social-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z"></path></svg>
            </a>
            <a href="#" class="social-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="20" x="2" y="2" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path><line x1="17.5" x2="17.51" y1="6.5" y2="6.5"></line></svg>
            </a>
             <a href="#" class="social-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect width="4" height="12" x="2" y="9"></rect><circle cx="4" cy="4" r="2"></circle></svg>
            </a>
          </div>
        </div>
      </div>
      <div class="footer-bottom">
        <p>&copy; 2025 TWEETIFY. All rights reserved.</p>
        <p>
          <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a>
        </p>
      </div>
    </footer>

    <script
      type="module"
      src="https://unpkg.com/@splinetool/viewer@1.9.91/build/spline-viewer.js"
    ></script>
    
    <script>
      function showClassifySection() {
        const classifySection = document.getElementById('classify-section');
        classifySection.classList.add('active');
        
        // Smooth scroll to the section
        setTimeout(() => {
          classifySection.scrollIntoView({behavior: 'smooth', block: 'center'});
        }, 100);
      }
      
      function resetForm() {
        document.getElementById('uploadForm').reset();
        document.getElementById('resultContainer').innerHTML = "";
        document.getElementById('loading').style.display = "none";
        
        // Add button animation when resetting
        const resetBtn = document.querySelector('.reset-btn');
        resetBtn.classList.add('pulse');
        setTimeout(() => {
          resetBtn.classList.remove('pulse');
        }, 500);
      }
      
      // Add event listener to show loading indicator when form is submitted
      // Add this script at the end of the HTML file, just before the closing </body> tag

document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('uploadForm');
  const loadingIndicator = document.getElementById('loading');
  const resultContainer = document.getElementById('resultContainer');
  
  if(form) {
    form.addEventListener('submit', function(e) {
      e.preventDefault(); // Prevent the default form submission
      
      loadingIndicator.style.display = 'block';
      
      // Create FormData object
      const formData = new FormData(this);
      
      // Create and configure the AJAX request
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/', true);
      xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
      
      // Handle the response
      xhr.onload = function() {
        if (xhr.status === 200) {
          try {
            const response = JSON.parse(xhr.responseText);
            resultContainer.innerHTML = response.result_html;
          } catch (e) {
            resultContainer.innerHTML = '<div class="result">Error processing response.</div>';
            console.error('Error parsing JSON:', e);
          }
        } else {
          resultContainer.innerHTML = '<div class="result">Error processing your request.</div>';
        }
        loadingIndicator.style.display = 'none';
      };
      
      // Handle errors
      xhr.onerror = function() {
        resultContainer.innerHTML = '<div class="result">Network error occurred.</div>';
        loadingIndicator.style.display = 'none';
      };
      
      // Send the form data
      xhr.send(formData);
    });
  }
  
  // Button hover animations (keeping the existing code)
  const buttons = document.querySelectorAll('.submit-btn, .reset-btn');
  buttons.forEach(button => {
    button.addEventListener('mouseover', function() {
      this.style.transition = 'all 0.3s ease';
    });
  });
});
    </script>
  </body>
</html>

    
    """, mimetype='text/html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)