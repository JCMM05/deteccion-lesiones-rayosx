# 🦴 Detección de Lesiones Óseas en Radiografías

Sistema de apoyo diagnóstico que detecta anomalías óseas en radiografías usando CNN + Grad-CAM + IA Especialista (LLaMA 4).

## 🛠️ Tecnologías

- **Modelo:** EfficientNetB0 con Transfer Learning
- **Visualización:** Grad-CAM (mapas de calor)
- **IA Especialista:** Groq + LLaMA 4
- **Interfaces:** Gradio y Streamlit
- **Dataset:** MURA (Stanford) — 36,812 imágenes

## 📊 Métricas

- **Accuracy:** 71%
- **AUC-ROC:** 0.773
- **F1-Score:** 0.71

## 🚀 Cómo ejecutar

```bash
# App Gradio
python app_gradio.py

# App Streamlit
streamlit run app_streamlit.py
```

## ⚠️ Aviso

Este sistema es una herramienta de apoyo diagnóstico. No reemplaza el criterio de un médico especialista.
