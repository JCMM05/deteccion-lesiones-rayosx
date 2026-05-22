import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import cv2
import os
from dotenv import load_dotenv
load_dotenv()
import base64
import io

# ── Configuración ──────────────────────────────────────────
MODEL_PATH = r'C:\Users\WinterOS\deteccion-lesiones-rayosx\models\best_model.pth'
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Cargar modelo ──────────────────────────────────────────
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

# ── Transformaciones ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Grad-CAM ───────────────────────────────────────────────
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

# ── Predicción principal ───────────────────────────────────
def predict(image):
    if image is None:
        return None, "Por favor sube una imagen"

    img = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    label = "🔴 ANORMAL" if prob > 0.5 else "🟢 NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob

    # Grad-CAM
    img_tensor_grad = transform(img).unsqueeze(0).to(device).requires_grad_(True)
    cam = generate_gradcam(img_tensor_grad, model)

    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Groq
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        # Convertir imagen a base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Eres un radiólogo especialista. El modelo CNN clasificó esta radiografía como {'ANORMAL' if prob > 0.5 else 'NORMAL'} con {confidence:.1%} de confianza. Analiza la imagen y da un veredicto clínico breve en español."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        especialista_text = response.choices[0].message.content
    except Exception as e:
        especialista_text = f"Especialista IA no disponible: {str(e)}"

    resultado = f"""
## Resultado del Modelo CNN
**Diagnóstico:** {label}
**Confianza:** {confidence:.1%}

---
##  🤖 Diagnóstico Asistido por IA
{especialista_text}
"""
    return overlay, resultado

# ── Interfaz Gradio ────────────────────────────────────────
with gr.Blocks(title="Detección de Lesiones Óseas") as app:
    gr.Markdown("# 🦴 Detección de Lesiones Óseas en Radiografías")
    gr.Markdown("Sube una radiografía o toma una foto para detectar anomalías óseas.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                sources=["upload", "webcam"],
                label="📷 Radiografía",
                type="numpy"
            )
            btn = gr.Button("🔍 Analizar", variant="primary")

        with gr.Column():
            output_cam = gr.Image(label="🔥 Mapa de Calor (Grad-CAM)")
            output_text = gr.Markdown(label="📋 Resultado")

    btn.click(fn=predict, inputs=input_image, outputs=[output_cam, output_text])

if __name__ == "__main__":
    app.launch()