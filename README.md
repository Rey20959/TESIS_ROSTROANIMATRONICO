# TESIS_ROSTROANIMATRONICO
Este es el trabajo de graduación, así siendo la integración de ciertos programas anteriores con mejoras. Así teniendo la detección de emociones junto con un chatbot incorporado a la inteligencia artificial.


Dentro del proyecto se encontraran distintos documentos, los cuales se tendrán que descargar ciertas librerías, las cuales son las sigueintes:
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

ASEGURARSE QUE ESTAS ESTEN INSTALADAS CORRECTAMENTE. 

Luego de que están instaladas correctamente, se debe de cambiar la dirección de estos documentos:

# Configuración de modelos para detección de emociones
face_cascade_path = r"C:\Users\juane\OneDrive\Desktop\TESIS\TESIS\Trabajo_Fisico\Prueba AI\Proyecto_Web_ChatGPT_Emociones\haarcascade.xml"
emotion_model_path = r"C:\Users\juane\OneDrive\Desktop\TESIS\TESIS\Trabajo_Fisico\Prueba AI\Proyecto_Web_ChatGPT_Emociones\EmotionDetectionModelElu5.h5"


SON 2 archivos, los cuales se encuentran en la carpeta, así que solo es cambiar la dirección de cada archivo que tiene el nombre que aparece ahí.


Por último se debe de correr el archivo **app.py**, este será el encargado de correr toda la página.

Luego que se corra, saldrá un enlace. El enlace es el siguiente: http://127.0.0.1:5000/

Se debe de poner ese enlace en google chrome (Recomendable) para que así pueda empezar a funcionar. 

Se debe de dar total acceso al buscador para que utilice la cámara, y el micrófono. 

Y eso sería todo. 
Cualquier consulta quedo a la orden. 
