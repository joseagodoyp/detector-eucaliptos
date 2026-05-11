from ultralytics import YOLO

print("Cargando modelo...")
model = YOLO("yolov8n.pt")
print("✅ Modelo cargado correctamente")

print("Clases que puede detectar:", model.names)
print("🎉 Todo funciona! Tu IA está lista.")