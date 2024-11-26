from flask import Flask, render_template, request, jsonify, redirect, send_file
import openai
import json
import os
import cv2
import base64
import numpy as np
from keras.utils import img_to_array
from tensorflow.keras.models import load_model
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Configuración de modelos para detección de emociones
face_cascade_path = r"C:\Users\juane\OneDrive\Desktop\TESIS\TESIS\Trabajo_Fisico\Prueba AI\Proyecto_Web_ChatGPT_Emociones\haarcascade.xml"
emotion_model_path = r"C:\Users\juane\OneDrive\Desktop\TESIS\TESIS\Trabajo_Fisico\Prueba AI\Proyecto_Web_ChatGPT_Emociones\EmotionDetectionModelElu5.h5"

# Cargar los modelos
face_cascade = cv2.CascadeClassifier(face_cascade_path)
emotion_model = load_model(emotion_model_path)
emotion_labels = ['Enojo', 'Disgusto', 'Miedo', 'Alegria', 'Neutral', 'Triste', 'Sorpresa']


# Funciones para cargar y guardar información adicional desde el archivo JSON

def load_json():
    file_path = os.path.join(os.getcwd(), 'uvg_departamento_ingenieria.json')  # Ruta absoluta
    try:
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Error: El archivo {file_path} no existe.")
            return {}  # Retorna un diccionario vacío si no existe
    except Exception as e:
        print(f"Error al leer el archivo JSON: {e}")
        return {}

def save_json(data):
    file_path = os.path.join(os.getcwd(), 'uvg_departamento_ingenieria.json')  # Ruta absoluta
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error al guardar el archivo JSON: {e}")

# Configuración de OpenAI (deberías considerar mover la clave API a una variable de entorno)
openai.api_key = 'sk-proj-1rW1bTeKWWmbJKga3GRQT3BlbkFJl0KRjCrNbbilAtaleJuc'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_connection')
def check_connection():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Este es un mensaje de prueba."}]
        )
        # Si la conexión es exitosa, subir el archivo JSON
        upload_data()
        return jsonify({'connected': True})
    except Exception as e:
        print(f"Error al conectar con OpenAI: {e}")
        return jsonify({'connected': False})

@app.route('/upload_data', methods=['POST'])
def upload_data():
    # Simulación de la lógica para manejar la subida de datos
    data_to_upload = load_json()
    # Aquí puedes implementar la lógica para subir los datos si es necesario
    print("Datos subidos exitosamente")
    return jsonify({'status': 'success'})

@app.route('/chat', methods=['POST'])
def chat():
    uvg_data = load_json()  # Recargar los datos para asegurar que estén actualizados
    user_input = request.form['text']
    emotion = request.form['emotion']
    response = get_chatgpt_response(user_input, emotion, uvg_data)
    return jsonify({'response': response})

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    """
    Endpoint para detección de emociones. Recibe una imagen en formato base64,
    la analiza y devuelve la emoción detectada.
    """
    data = request.json
    frame_data = data.get('frame')
    
    # Decodificar imagen base64
    encoded_data = frame_data.split(',')[1]
    frame = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({"emotion": "Ninguna cara detectada"})

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        predictions = emotion_model.predict(roi)[0]
        emotion = emotion_labels[np.argmax(predictions)]
        return jsonify({"emotion": emotion})

import tiktoken  # Importa tiktoken para calcular tokens

def get_chatgpt_response(text, emotion, uvg_data):
    # Generar contexto adicional desde el archivo JSON
    context = ""
    if "Universidad" in uvg_data:
        universidad_info = uvg_data["Universidad"]
        context = f"\n\nContexto adicional: {universidad_info['Nombre']} está ubicada en {universidad_info['Ubicación']}. " \
                  f"Fue fundada en {universidad_info['Fundación']} y su lema es '{universidad_info['Lema']}'."
    
    if "Informacion Adicional" in uvg_data:
        additional_info = "\n".join(uvg_data["Informacion Adicional"])
        context += f"\nInformación adicional: {additional_info}"
    
    prompt = f"Por favor responde de manera breve y clara. La emoción detectada es {emotion}. Responde a: {text}{context}"
    
    # Calcular dinámicamente los tokens disponibles
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    prompt_tokens = len(encoding.encode(prompt))
    max_tokens_limit = 4096 - prompt_tokens - 50  # Reservar espacio para tokens de respuesta y seguridad

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un chatbot digital universitario, amigable y capaz de detectar emociones."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens_limit,
        temperature=0.3,
        stop=["\n"]
    )
    return response['choices'][0]['message']['content'].strip()

@socketio.on('message')
def handle_message(data):
    uvg_data = load_json()  # Recargar los datos para asegurar que estén actualizados
    user_message = data['message']
    emotion = data['emotion']
    response = get_chatgpt_response(user_message, emotion, uvg_data)
    emit('response', {'message': response, 'emotion': emotion})

@app.route('/educa_ia', methods=['GET', 'POST'])
def educa_ia():
    uvg_data = load_json()  # Recargar los datos para asegurar que estén actualizados
    if request.method == 'POST':
        new_info = request.form['new_info']
        if "Informacion Adicional" not in uvg_data:
            uvg_data["Informacion Adicional"] = []
        uvg_data["Informacion Adicional"].append(new_info)
        save_json(uvg_data)
        return render_template('educa_ia.html', success=True)
    return render_template('educa_ia.html', success=False)

@app.route('/deteccion_emociones')
def deteccion_emociones():
    return render_template('deteccion_emociones.html')

@app.route('/informacion_almacenada')
def informacion_almacenada():
    uvg_data = load_json()

    # Usar .get para evitar el error si la clave 'Universidad' no existe
    universidad_info = uvg_data.get('Universidad', None)

    return render_template('informacion_almacenada.html', data=uvg_data, universidad=universidad_info)

@app.route('/descargar_informacion')
def descargar_informacion():
    file_path = os.path.join(os.getcwd(), 'uvg_departamento_ingenieria.json')  # Ruta absoluta
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "Error: El archivo no existe", 404

@app.route('/informacion_extra')
def informacion_extra():
    return redirect("https://www.uvg.edu.gt")

@app.route('/conversacion_vivo')
def conversacion_vivo():
    return render_template('conversacion_vivo.html')

if __name__ == "__main__":
    socketio.run(app, debug=True)



#Para correrlo python app.py

#y abrir esto en Google chrome http://127.0.0.1:5000/