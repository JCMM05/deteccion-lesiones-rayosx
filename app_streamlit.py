import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import cv2
import os
import base64
import io
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Configuración ──────────────────────────────────────────
MODEL_PATH = r'C:\Users\WinterOS\deteccion-lesiones-rayosx\models\best_model.pth'
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Cargar modelo ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(img_tensor, model):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    target_layer = model.features[-1]
    hook = target_layer.register_forward_hook(forward_hook)
    output = model(img_tensor)
    model.zero_grad()
    output.backward()
    hook.remove()

    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()
    return cam

# ── Página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de Lesiones Óseas",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2196F3);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #b3d4f5;
        margin: 5px 0 0 0;
        font-size: 0.95rem;
    }
    .result-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .disclaimer {
        background: #2d2d2d;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #ff9800;
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2196F3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🦴 Detección de Lesiones Óseas en Radiografías</h1>
    <p>Sistema de apoyo diagnóstico usando CNN + Grad-CAM + IA Especialista</p>
</div>
""", unsafe_allow_html=True)

# Layout principal
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Cargar Radiografía")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Radiografía cargada", width=280)
        analizar = st.button("🔍 Analizar Radiografía")
    else:
        st.info("👆 Sube una radiografía para comenzar el análisis")
        analizar = False

with col2:
    if uploaded_file and analizar:
        with st.spinner("🔄 Analizando radiografía..."):
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()

            label = "🔴 ANORMAL" if prob > 0.5 else "🟢 NORMAL"
            confidence = prob if prob > 0.5 else 1 - prob
            color = "#ff4444" if prob > 0.5 else "#44ff44"

            img_tensor_grad = transform(image).unsqueeze(0).to(device).requires_grad_(True)
            cam = generate_gradcam(img_tensor_grad, model)
            img_np = np.array(image.resize((224, 224)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            # Groq
            try:
                client = Groq(api_key=GROQ_API_KEY)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Eres un radiólogo especialista. El modelo CNN clasificó esta radiografía como {'ANORMAL' if prob > 0.5 else 'NORMAL'} con {confidence:.1%} de confianza. Analiza la imagen y da un veredicto clínico breve en español."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            }
                        ]
                    }],
                    max_tokens=300
                )
                especialista_text = response.choices[0].message.content
            except Exception as e:
                especialista_text = f"IA no disponible: {str(e)}"

        st.markdown("### 🔥 Mapa de Calor (Grad-CAM)")
        st.image(overlay, width=280)

        st.markdown("### 📊 Resultados")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Diagnóstico", "ANORMAL" if prob > 0.5 else "NORMAL")
        with m2:
            st.metric("Confianza", f"{confidence:.1%}")

        st.progress(confidence)

        st.markdown("### 🤖 Diagnóstico Asistido por IA")
        st.markdown(f"""
<div class="result-card">
{especialista_text}
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="disclaimer">
⚠️ <strong>Aviso:</strong> Este sistema es una herramienta de apoyo diagnóstico. 
No reemplaza el criterio de un médico especialista. Consulte siempre a un profesional de la salud.
</div>
""", unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("### 📋 Instrucciones")
        st.markdown("""
        1. 📤 Sube una imagen de radiografía (PNG, JPG)
        2. 🔍 Click en **Analizar Radiografía**
        3. 🔥 Ve el mapa de calor Grad-CAM
        4. 📊 Revisa el diagnóstico del modelo CNN
        5. 🤖 Lee el análisis del especialista IA
        """)