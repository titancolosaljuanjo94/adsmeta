
import io
import os
import re
import zipfile
import time
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple

# --- OpenAI ---
try:
    from openai import OpenAI
    openai_client = OpenAI()
except Exception:
    openai_client = None

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(page_title="Generador de Metadescripciones", page_icon="🧩", layout="wide")

st.title("Generador de Metadescripciones (4 por carrera)")
st.caption("Sube tu ZIP de PDFs o configura una ZIP_URL en Secrets. Elige la carrera y el tono, y genera 4 metadescripciones optimizadas para Ads.")

# === Rango estricto de caracteres para cada metadescripción ===
MIN_LEN = 135
MAX_LEN = 155

# =========================
# UTILIDADES
# =========================
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_zip_from_url(url: str) -> bytes:
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def clean_name(filename: str) -> str:
    name = re.sub(r"\.pdf$", "", filename, flags=re.I)
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().title()
    return name

def extract_pdf_text_from_zip(zb: bytes, inner_path: str, max_pages: int = 8) -> str:
    """
    Extrae texto de las primeras páginas del PDF (para contexto rápido).
    Requiere PyPDF2. Si falla, devuelve cadena vacía.
    """
    try:
        import PyPDF2
    except ImportError:
        return ""

    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        with zf.open(inner_path) as f:
            reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
            pages_to_read = min(len(reader.pages), max_pages)
            text = []
            for i in range(pages_to_read):
                try:
                    text.append(reader.pages[i].extract_text() or "")
                except Exception:
                    pass
            return "\n".join(text).strip()

def list_pdfs(zb: bytes) -> List[str]:
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        pdfs = [p for p in zf.namelist() if p.lower().endswith(".pdf")]
    return sorted(pdfs)

# =========================
# SIDEBAR: CONTROLES
# =========================

# API key desde entorno o Secrets
# =========================
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key  # para que el SDK la use
    api_configured = True
else:
    api_configured = False

with st.sidebar:
    st.subheader("Configuración")

    model = st.selectbox("Modelo", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"])
    tone = st.selectbox(
        "Tono de comunicación",
        ["Neutro", "Cercano", "Técnico", "Aspiracional", "Institucional", "Orientado a Conversión"]
    )

    generation_mode = st.radio("Generar para", ["Solo 1 carrera", "Varias carreras"])

    # 👇 línea informativa (ya sin uploader ni input de URL)
st.caption("Fuente: ZIP preconfigurado (ZIP_URL en Secrets).")
    st.markdown("---")
    st.caption("Estado de API:")
    if api_configured:
        st.success("API configurada")
    else:
        st.warning("Falta OPENAI_API_KEY (añádela en Settings → Secrets)")



# CARGA DEL ZIP (automática con ZIP_URL desde Secrets/entorno)
# =========================
zip_url = os.environ.get("ZIP_URL") or st.secrets.get("ZIP_URL", "")
if not zip_url:
    st.error("Falta ZIP_URL en Secrets. Ve a Settings → Secrets y agrega ZIP_URL con tu enlace de Google Drive.")
    st.stop()

with st.spinner("Descargando ZIP desde ZIP_URL..."):
    try:
        zip_bytes = fetch_zip_from_url(zip_url)
    except Exception as e:
        st.error(f"No se pudo descargar el ZIP desde ZIP_URL: {e}")
        st.stop()


pdf_paths = list_pdfs(zip_bytes)
if not pdf_paths:
    st.error("No se encontraron PDFs dentro del ZIP.")
    st.stop()

# Mapa Carrera -> path interno en el ZIP
options = [(clean_name(os.path.basename(p)), p) for p in pdf_paths]

# =========================
# UI DE SELECCIÓN
# =========================
left, right = st.columns([1, 1])

with left:
    if generation_mode == "Solo 1 carrera":
        selected = st.selectbox("Selecciona la carrera", [o[0] for o in options])
        selected_paths = [dict(options)[selected]]
    else:
        picks = st.multiselect("Selecciona una o más carreras", [o[0] for o in options], max_selections=10)
        selected_paths = [dict(options)[p] for p in picks] if picks else []

with right:
    st.write("Vista previa del ZIP")
    st.dataframe(pd.DataFrame(
        [{"Carrera": n, "Archivo": p} for (n, p) in options],
    ), use_container_width=True, hide_index=True)

# =========================
# PROMPTS
# =========================
def build_prompt(career_name: str, context_text: str, tone: str) -> str:
    return f"""
Eres un especialista en contenido experto en **Google Ads** y en generación de campañas de Google Search. Genera **4 metadescripciones distintas** para la carrera que el usuario elija **{career_name}**.
Usa la información del brochure que hay por cada una de las carreras (si es útil); NO inventes datos:

[CONTEXTO]
{context_text[:4000]}

**Requisitos estrictos:**
- Debes generar 4 metadescripciones, no menos no más por cada carrera.
- Cada metadescripción debe tener entre **{MIN_LEN} y {MAX_LEN}** caracteres (ni menos ni más).
- Si el texto queda fuera de este rango, reescríbelo hasta cumplirlo.
- Evita comillas, emojis y mayúsculas excesivas.
- Frases claras, orientadas a clic y a valor.
- Tono: **{tone}**.
- Incluye beneficios/ventajas concretas cuando sea posible.
- No repitas exactamente la misma estructura en las 4.
- Puedes empezar con mensajes tales como: Estudia la carrera de **{career_name}**, Comienza tu camino profesional, Vuélvete un experto, Alcanza tus sueños.
- Puedes terminar las metadescripciones con un call to action, que invite al usuario a conocer más sobre nuestra carrera. Por ejemplo: ¡Conoce más aquí!, ¡Inscríbete hoy! y ¡Matrículate hoy!.

Devuelve únicamente una lista con 4 líneas, una por cada metadescripción, sin numerarlas.
""".strip()

def call_openai_generate(prompt: str, model: str) -> List[str]:
    """
    Llama a la API de OpenAI (chat.completions) y devuelve hasta 4 líneas.
    """
    if not openai_client:
        from openai import OpenAI
        client = OpenAI()
    else:
        client = openai_client

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un experto en redacción de metadescripciones para Google Ads."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    text = (resp.choices[0].message.content or "").strip()
    lines = [l.strip().lstrip("•- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:4]

# =========================
# ACCIONES
# =========================
def generate_for_one(path_in_zip: str, career_label: str) -> Tuple[str, List[str]]:
    # 1) Contexto del PDF
    ctxt = extract_pdf_text_from_zip(zip_bytes, path_in_zip, max_pages=8)

    # 2) Prompt
    prompt = build_prompt(career_label, ctxt, tone)

    # 3) Llamada a OpenAI
    variants = call_openai_generate(prompt, model)

    # 4) Forzar rango 135–155: recortar > MAX_LEN y descartar < MIN_LEN
    cleaned: List[str] = []
    for v in variants:
        v = (v or "").strip()
        if len(v) > MAX_LEN:
            v = v[:MAX_LEN].rstrip()
        if MIN_LEN <= len(v) <= MAX_LEN:
            cleaned.append(v)

    return career_label, cleaned[:4]

st.markdown("### Generación")
col_a, col_b = st.columns([1, 2])

with col_a:
    do_generate = st.button("Generar", type="primary", use_container_width=True)

with col_b:
    st.info("Consejo: Si el PDF es muy pesado o con mucho gráfico, el texto extraído puede ser limitado. El prompt se adapta igualmente.")

results: Dict[str, List[str]] = {}
if do_generate:
    if generation_mode == "Varias carreras":
        st.warning("El modo 'Varias carreras' está deshabilitado en este prototipo para controlar costos. Usa 'Solo 1 carrera'.")
        st.stop()

if do_generate:
    if not api_key:
        st.error("Falta OPENAI_API_KEY. Agrega tu clave en Settings → Secrets.")
        st.stop()

    targets = selected_paths if selected_paths else [options[0][1]]
    labels_map = {p: n for (n, p) in options}

    with st.spinner("Generando metadescripciones..."):
        for p in targets:
            name = labels_map[p]
            try:
                label, metas = generate_for_one(p, name)
                results[label] = metas
                time.sleep(0.2)  # leve pausa para UI estable
            except Exception as e:
                st.warning(f"{name}: error al generar → {e}")

    if results:
        st.success("¡Listo! Metadescripciones generadas.")
        rows = []
        for career, metas in results.items():
            for i, m in enumerate(metas, start=1):
                rows.append({"Carrera": career, "Variante": i, "Metadescripción": m, "Caracteres": len(m)})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", csv_bytes, file_name="metadescripciones.csv", mime="text/csv")
