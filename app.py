import streamlit as st
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Detector de Eucaliptos", page_icon="tree")
st.title("Detector de Eucaliptos con IA")
st.write("Subi una foto y la IA detectara los objetos automaticamente.")

@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")

modelo = cargar_modelo()

archivo = st.file_uploader("Subi una foto", type=["jpg", "jpeg", "png"])

if archivo is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(archivo.getbuffer())
        ruta_tmp = tmp.name
    
    st.image(archivo, caption="Imagen original", use_container_width=True)
    
    with st.spinner("Analizando..."):
        resultados = modelo(ruta_tmp)
    
    imagen_resultado = resultados[0].plot()
    st.image(imagen_resultado, caption="Detecciones", use_container_width=True)
    
    cantidad = len(resultados[0].boxes)
    st.success(f"Se detectaron {cantidad} objetos")
    
    os.unlink(ruta_tmp)