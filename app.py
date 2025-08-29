from flask import Flask, render_template, request, jsonify
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from collections import Counter

# Define Flask app
app = Flask(__name__)

# Define class names
CLASS_NAMES = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "EDM",
    4: "Hip-Hop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

# Function to split audio
def split_audio(file_path, chunk_duration=10):
    y, sr = librosa.load(file_path, sr=None)
    chunk_length = chunk_duration * sr
    return [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)], sr

# Function to create Mel spectrogram
def create_mel_spectrogram(chunk, sr, output_folder, file_prefix, segment_index):
    os.makedirs(output_folder, exist_ok=True)
    mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
    plt.axis('off')
    temp_image_path = os.path.join(output_folder, f"temp_{file_prefix}_{segment_index}.png")
    plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close()
    with Image.open(temp_image_path) as img:
        img = img.resize((512, 512))
        final_path = os.path.join(output_folder, f"{file_prefix}_mel_spectrogram_{segment_index}.png")
        img.save(final_path)
    os.remove(temp_image_path)

# Function to process and classify
def process_and_classify(audio_path, model_path):
    model = load_model(model_path)
    output_dir = "output_spectrograms"
    os.makedirs(output_dir, exist_ok=True)
    chunks, sr = split_audio(audio_path)
    all_probabilities = []
    all_class_indices = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < sr * 10:
            continue
        file_prefix = f"chunk_{i + 1}"
        create_mel_spectrogram(chunk, sr, output_dir, file_prefix, i + 1)
        mel_img_path = os.path.join(output_dir, f"{file_prefix}_mel_spectrogram_{i + 1}.png")
        mel_img = Image.open(mel_img_path).convert('RGB')
        img_array = np.array(mel_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        probabilities = model.predict(img_array)
        predicted_index = np.argmax(probabilities[0])
        all_probabilities.append(probabilities[0])
        all_class_indices.append(predicted_index)
    class_counts = Counter(all_class_indices)
    total_chunks = sum(class_counts.values())
    overall_percentages = {
        CLASS_NAMES[cls]: (count / total_chunks) * 100 for cls, count in class_counts.items()
    }
    sorted_percentages = sorted(overall_percentages.items(), key=lambda x: x[1], reverse=True)
    return sorted_percentages, sorted_percentages[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    h5_model_path = "model_5_90(76.8, 1.7).h5"  
    results, top_class = process_and_classify(file_path, h5_model_path)
    return jsonify({"top_genre": top_class, "distribution": results})

if __name__ == '__main__':
    app.run(debug=True)
