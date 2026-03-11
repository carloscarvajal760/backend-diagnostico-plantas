from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import os
import json

# Inicializar la app
app = Flask(__name__, static_folder='../frontend_diagnostico/build', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuración de rutas
MODEL_PATH = 'modelo/modelo_finall.keras'  # Asegúrate que el nombre coincida exactamente
CLASSES_PATH = 'modelo/class_namee.json'

# --- FUNCIÓN PARA RECONSTRUIR EL MODELO ---
# Esto soluciona el error: ValueError: Layer "dense" expects 1 input(s)...
def build_model_structure(num_classes):
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# --- CARGA DEL MODELO Y CLASES ---
# Primero cargamos las clases para saber el tamaño de la última capa
try:
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    print(f"Clases cargadas correctamente: {len(class_names)} categorías.")
except Exception as e:
    print(f"Error crítico cargando JSON de clases: {e}")
    class_names = []

# Intentamos cargar el modelo de forma robusta
try:
    # Intento 1: Carga directa (estándar de Keras 3)
    model = load_model(MODEL_PATH)
    print("Modelo cargado exitosamente mediante load_model.")
except Exception as e:
    print(f"Aviso: load_model falló ({e}). Intentando reconstrucción manual...")
    # Intento 2: Reconstrucción de arquitectura + carga de pesos
    if class_names:
        model = build_model_structure(len(class_names))
        model.load_weights(MODEL_PATH)
        print("Modelo reconstruido y pesos aplicados con éxito.")
    else:
        print("Error: No se pudo reconstruir el modelo porque no hay clases definidas.")

# --- PROCESAMIENTO DE IMAGEN ---
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    # Importante: Usar float32 para coincidir con la precisión del entrenamiento
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- RUTAS DE LA API ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío.'}), 400

    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    try:
        img_array = preprocess_image(temp_path)
        predictions = model.predict(img_array)

        class_id = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = "Clase desconocida"

        return jsonify({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': round(confidence, 4),
            'status': 'success'
        })
    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- SERVIR FRONTEND (REACT) ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Usar host='0.0.0.0' por si quieres probar desde tu celular en la misma red WiFi
    app.run(host='0.0.0.0', port=5000, debug=True)