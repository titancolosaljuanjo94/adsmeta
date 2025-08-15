# Metadescription Tool – Streamlit Cloud Ready

Esta app genera **4 metadescripciones por carrera** usando OpenAI y PDFs dentro de un ZIP.
Funciona en **Streamlit Cloud** con dos modos de entrada:

1) **Subir ZIP** desde la interfaz (recomendado).
2) **ZIP_URL** en Secrets para descargar el ZIP automáticamente.

## Despliegue en Streamlit Cloud

1. Sube estos archivos a tu repositorio en GitHub.
2. En **share.streamlit.io** crea el deploy con:
   - **Main file path:** `streamlit_app.py`
3. En **Settings → Secrets** agrega:
```
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
# Opcional si no subirás ZIP manualmente:
ZIP_URL = "https://tuservidor.com/carreras.zip"
```
4. Deploy / Restart.

## Uso
- **Opción Subir ZIP:** Arrastra tu archivo `.zip` con PDFs. La app listará las carreras y generará 4 metadescripciones.
- **Opción ZIP_URL:** Define `ZIP_URL` en Secrets. La app descargará el ZIP automáticamente.

> Si tus PDFs son escaneados (solo imagen), el extractor no encontrará texto. Para esto necesitarías OCR en una futura versión.
