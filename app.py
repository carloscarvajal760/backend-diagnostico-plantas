import os
import json
import gc
import numpy as np
import tensorflow as tf

# CONFIGURACIÓN DE BAJO CONSUMO
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU') # Forzar CPU

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
# CORS agresivo para evitar el error de consola
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# 2. Rutas Absolutas (Vital para que Render encuentre los archivos)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo', 'modelo_finall.keras')
CLASSES_PATH = os.path.join(BASE_DIR, 'modelo', 'class_namee.json')

# 3. Función para reconstruir la estructura si falla load_model
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

# 4. Carga de Clases y Modelo al arrancar el servidor
class_names = []
model = None

try:
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"✅ Clases cargadas: {len(class_names)}")
    else:
        print(f"❌ Error: No se encontró el archivo en {CLASSES_PATH}")
except Exception as e:
    print(f"❌ Error cargando JSON: {e}")

try:
    if os.path.exists(MODEL_PATH):
        # Intento 1: Carga directa
        try:
            model = load_model(MODEL_PATH)
            print("✅ Modelo cargado con load_model")
        except Exception:
            # Intento 2: Reconstrucción (por si hay error de capas en Keras 3)
            if class_names:
                model = build_model_structure(len(class_names))
                model.load_weights(MODEL_PATH)
                print("✅ Modelo reconstruido con éxito")
    else:
        print(f"❌ Error: No se encontró el modelo en {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {e}")

# 5. Rutas de la API
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "mensaje": "Backend de Diagnóstico de Plantas listo",
        "modelo_cargado": model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'El modelo no está cargado en el servidor.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo no seleccionado.'}), 400

    # Guardar imagen temporalmente
    temp_path = os.path.join(BASE_DIR, 'temp_predict.jpg')
    file.save(temp_path)

    try:
        # Preprocesamiento
        img = Image.open(temp_path).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        predictions = model.predict(img_array)
        class_id = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        resultado = {
            'class_id': class_id,
            'class_name': class_names[class_id] if class_id < len(class_names) else "Desconocido",
            'confidence': round(confidence, 4),
            'status': 'success'
        }
        return jsonify(resultado)

    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Render usa Gunicorn, pero esto sirve para pruebas locales
    app.run(host='0.0.0.0', port=5000)