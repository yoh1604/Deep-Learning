from ultralytics import YOLO
import inference
from roboflow import Roboflow

export ROBOFLOW_API_KEY=rf_FhGkkpYVFLcaDr6qmiBEmZmP9Vz1
rf = Roboflow(api_key="rf_FhGkkpYVFLcaDr6qmiBEmZmP9Vz1")
model = inference.load_roboflow_model("yolov8n-640")
results = model.infer(image="YOUR_IMAGE.jpg")

# Training
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
# Evaluasi model di validation set
metrics = model.val()

# Prediksi di gambar
results = model.predict(source='path_to_image.jpg', save=True, conf=0.25)
