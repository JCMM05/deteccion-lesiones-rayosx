# Detección de Lesiones Óseas en Radiografías

Sistema inteligente de apoyo diagnóstico que detecta anomalías óseas en radiografías usando Deep Learning.

## Objetivo

Desarrollar una aplicación capaz de detectar anomalías óseas en radiografías de rayos X, utilizando CNN con Transfer Learning, mapas de calor Grad-CAM y un especialista IA para apoyar procesos de diagnóstico médico.

## Demo en Vivo

[Ver aplicación en Hugging Face Spaces](https://huggingface.co/spaces/JCMM05/deteccion-lesiones-rayosx)

## Tecnologías

- **Modelo:** EfficientNetB0 con Transfer Learning (PyTorch)
- **Visualización:** Grad-CAM (mapas de calor)
- **IA Especialista:** Groq + LLaMA 4
- **Interfaces:** Gradio y Streamlit
- **Dataset:** MURA Stanford — 36,812 imágenes

## Métricas del Modelo

| Métrica   | Valor |
| --------- | ----- |
| Accuracy  | 71%   |
| AUC-ROC   | 0.773 |
| F1-Score  | 0.71  |
| Precision | 0.71  |
| Recall    | 0.71  |

## Estructura del Proyecto

    deteccion-lesiones-rayosx/
    ├── notebooks/
    │   ├── 01_EDA.ipynb
    │   └── 02_Entrenamiento.ipynb
    ├── app_gradio.py
    ├── app_streamlit.py
    ├── requirements.txt
    └── README.md

## Instalación Local

    git clone https://github.com/JCMM05/deteccion-lesiones-rayosx.git
    cd deteccion-lesiones-rayosx
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Cómo Ejecutar

    # App Gradio
    python app_gradio.py

    # App Streamlit
    streamlit run app_streamlit.py

## Variables de Entorno

Crea un archivo .env con:
GROQ_API_KEY=tu_key_de_groq

## Ejemplos de Uso

1. Abre la app en el navegador
2. Sube una radiografía o usa la cámara web
3. Click en Analizar
4. Ve el mapa de calor Grad-CAM
5. Lee el diagnóstico del modelo CNN y el especialista IA

## Limitaciones

- El modelo fue entrenado con radiografías de extremidades superiores (MURA dataset)
- Accuracy del 71% — no reemplaza el criterio de un médico especialista
- Mejor rendimiento en muñecas y hombros por mayor representación en el dataset

## Dataset

MURA Stanford — Dataset de radiografías musculoesqueléticas con 40,561 imágenes etiquetadas por radiólogos de Stanford.
https://stanfordmlgroup.github.io/competitions/mura/

## Aviso Médico

Este sistema es una herramienta de apoyo diagnóstico. No reemplaza el criterio de un médico especialista. Consulte siempre a un profesional de la salud.
