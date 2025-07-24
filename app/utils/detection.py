import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from models.fast_rcnn.fast_rcnn import FastRCNNModel
import torch
from PIL import Image, ImageDraw



MODELS = {
    "butterfly": "runs/detect/inat_yolo_butterflies/weights/best.pt",
    "bee": "runs/detect/inat_yolo_bees/weights/best.pt",
    "fast_rcnn": "app/models/fast_rcnn/animal_classifier.pth",
    "chatbot": "app/models/hugging_face_t5/species-t5-finetuned/checkpoint-360"
}

CLASS_NAMES = {
    "butterfly": ["Papilio glaucus", "Danaus plexippus", "Vanessa atalanta", 
                "Heliconius charithonia", "Battus philenor", "Eurytides marcellus",
                "Speyeria cybele", "Junonia coenia", "Colias eurytheme", "Anartia jatrophae"],
    "bee": ["Osmia cornuta"],
    "fast_rcnn": ["cheetah", "fox", "hyena", "lion", "tiger", "wolf"]
}

def load_model(model_type, model_path):
    model_path = Path(MODELS[model_type])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if model_type == "fast_rcnn":
        model = FastRCNNModel()
        model.load_weights(str(model_path))
        return model
    else:
        model = YOLO(str(model_path))
        model.classes = CLASS_NAMES[model_type]
        return model

def predict_image(model, image, conf=0.25, imgsz=640):
    try:
        if isinstance(model, FastRCNNModel):
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image)
            else:
                pil_img = image

            prediction = model.predict(pil_img)

            annotated_img = pil_img.copy()
            draw = ImageDraw.Draw(annotated_img)
            draw.text((10, 10), f"{prediction['class']} ({prediction['confidence']:.2f})", fill="red")

            annotated_np = np.array(annotated_img)

            return [prediction], annotated_np

        else:
            results = model.predict(image, conf=conf, imgsz=imgsz)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class": model.classes[cls_id] if model.classes else str(cls_id),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
            annotated_img = results.plot() 
            return detections, annotated_img

    except Exception as e:
        raise RuntimeError(f"FastRCNNModel.predict() failed: {e}")

