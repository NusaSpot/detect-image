import PIL
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from io import BytesIO
import requests
import numpy as np
from google.cloud import storage
import os
 
app = Flask(__name__)
# model = tf.keras.models.load_model('image_classifier.h5')
bucket_name = 'nutricare'
model_filename = 'image_classifier.h5'
storage_client = storage.Client.from_service_account_json('cloud-storage.json')
 
temp_model_path = 'download/temp_model.h5'
 
# Memeriksa apakah file sudah ada
if not os.path.exists(temp_model_path):
    temp_model_directory = 'download'
    if not os.path.exists(temp_model_directory):
        os.makedirs(temp_model_directory)
    # Jika belum ada, lakukan pengunduhan
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_filename)
 
    model_file = BytesIO()
    blob.download_to_file(model_file)
    model_file.seek(0)
 
    # Menyimpan file ke lokasi lokal
    with open(temp_model_path, 'wb') as temp_model_file:
        temp_model_file.write(model_file.read())
 
# Load model using TensorFlow
model = tf.keras.models.load_model(temp_model_path)
app.config["ALLOWED_EXTENSIONS"] = set(['jpg', 'png', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
 
class_indices = np.load('class_dataset.npy', allow_pickle=True).item()
class_names = {v: k for k, v in class_indices.items()}
 
@app.route('/predict-single-url', methods=['POST'])
def predict_single_url():
    img_url = request.form.get('imgUrl')
 
    if img_url:
        # Mengambil gambar dari URL
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
 
        # Preprocess gambar dan lakukan prediksi
        x = img_to_array(img.resize((150, 150))) / 255.
        x = np.expand_dims(x, axis=0)
 
        prediction = model.predict(x)
 
        top_class = np.argmax(prediction)
 
        top_prob = prediction[0][top_class]
        top_class_name = class_names[top_class]
 
        result = {
            'input': img_url,
            'category': top_class_name,
            'prediction': f'{top_prob * 100:.2f}'
        }
 
        return jsonify({'status': 'SUCCESS', 'result': result}), 200
    else:
        return jsonify({'status': 'ERROR', 'message': 'No URL provided.'}), 400
 
@app.route('/predict-single', methods=['GET', 'POST'])
def predict_single():
    imgFile = request.files.get('imgFile[]')
 
    if imgFile and allowed_file(imgFile.filename):
        filename = secure_filename(imgFile.filename)    
        img_path = "./db/" + filename
        imgFile.save(img_path)
 
        img = load_img(img_path, target_size=(150, 150))
        x = img_to_array(img) / 255.
        x = np.expand_dims(x, axis=0)
 
        prediction = model.predict(x)
 
        top_class = np.argmax(prediction)
        top_prob = prediction[0][top_class]
 
        top_class_name = class_names[top_class]
 
        result = {
            'input': imgFile.filename,
            'category': top_class_name,
            'prediction': f'{top_prob * 100:.2f}'
        }
 
        return jsonify({'status': 'SUCCESS', 'result': result}), 200
    else:
        return jsonify({'status': 'ERROR', 'message': 'No file provided.'}), 400
 
@app.route('/predict-multiple', methods=['POST'])
def predict():
    files = request.files.getlist('imgFile[]')  # Menggunakan getlist untuk mengambil daftar file
 
    predictions = []
    if files:
        for imgFile in files:
            if imgFile and allowed_file(imgFile.filename):
                filename = secure_filename(imgFile.filename)    
                img_path = "./db/" + filename
                imgFile.save(img_path)
 
                img = load_img(img_path, target_size=(150, 150))
 
                x = img_to_array(img) / 255.
                x = np.expand_dims(x, axis=0)
 
                prediction = model.predict(x)
 
                top_class = np.argmax(prediction)
                top_prob = prediction[0][top_class]
 
                top_class_name = class_names[top_class]
 
                predictions.append({
                    'input': imgFile.filename,
                    'category': top_class_name,
                    'prediction': f'{top_prob * 100:.2f}'
                })
 
        return jsonify({'status': 'SUCCESS', 'predictions': predictions}), 200  # Format JSON untuk respons
    else:
        return jsonify({'status': 'ERROR', 'message': 'No file provided.'}), 400
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)