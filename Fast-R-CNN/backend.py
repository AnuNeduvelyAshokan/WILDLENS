from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime, select, func, desc, and_, delete
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
import os
import wikipediaapi
from typing import List, Dict
from pydantic import BaseModel
import json


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Database ---
DATABASE_URL = 'postgresql://postgres:admin@localhost:5432/animal_detection'
engine = create_engine(DATABASE_URL)
metadata = MetaData()

users_table = Table('users', metadata,
    Column('id', Integer, primary_key=True),
    Column('username', String, unique=True),
    Column('password', String)
)
predictions_table = Table('animal_predictions', metadata,
    Column('id', Integer, primary_key=True),
    Column('user_id', Integer),
    Column('predicted_class', String),
    Column('confidence', Float),
    Column('timestamp', DateTime)
)
metadata.create_all(engine)

# --- Wikipedia ---
wiki = wikipediaapi.Wikipedia(language='en', user_agent='SpeciesQA/1.0 (contact@example.com)')
def get_species_context(name):
    page = wiki.page(name)
    return page.summary if page.exists() else "No info found."

# --- distilgpt2 Article Generator ---
gpt_generator = pipeline("text-generation", model="distilgpt2")

def generate_educational_article(species_name, context=""):
    prompt = f"Write a short educational article about {species_name}. {context} Interesting facts:"
    result = gpt_generator(prompt, max_new_tokens=180, num_return_sequences=1)
    return result[0]['generated_text']

# --- T5 Wikipedia QA ---
t5_dir = Path(r"E:\modeling task\t5\species_t5_finetuned_clean").resolve()
t5_tokenizer = T5Tokenizer.from_pretrained(t5_dir.as_posix(), local_files_only=True)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_dir.as_posix(), local_files_only=True)

class T5Request(BaseModel):
    question: str
    species_name: str

@app.post("/t5_ask/")
async def t5_ask(request: T5Request):
    try:
        context = get_species_context(request.species_name)
        full_input = f"question: {request.question}  context: {context}"
        inputs = t5_tokenizer(full_input, return_tensors='pt')
        outputs = t5_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=120)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- generate_article endpoint ---
@app.post("/generate_article/")
async def generate_article(
    species_name: str = Form(...), 
    context: str = Form("")
):
    try:
        if not context:
            context = get_species_context(species_name)
        article = generate_educational_article(species_name, context)
        return {"article": article}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 56 * 56, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# Use the EXACT class order used in model training!
with open("mammal_classes.json") as f:
    mammal_classes = json.load(f)

cnn_model = SimpleCNN(num_classes=len(mammal_classes))
cnn_model.load_state_dict(torch.load("animal_classifier.pth", map_location="cpu"))
cnn_model.eval()



def preprocess_image(img_bytes):
    image = Image.open(img_bytes).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

import json

# Load class names from your JSON file (always do this! don't hardcode)
with open("mammal_classes.json", "r") as f:
    mammal_classes = json.load(f)

import json

# Load class names from your JSON file (always do this! don't hardcode)
with open("mammal_classes.json", "r") as f:
    mammal_classes = json.load(f)

@app.post("/predict_mammal/")
async def predict_mammal(user_id: int = Form(...), file: UploadFile = File(...)):
    # Read file bytes for PIL and transform
    image_tensor = preprocess_image(file.file)

    with torch.no_grad():
        logits = cnn_model(image_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_index = int(np.argmax(probs))
    species = mammal_classes[pred_index]
    confidence = float(probs[pred_index])

    # DEBUG: Print class mapping and probabilities
    print(f"Class mapping: {mammal_classes}")
    print(f"Probabilities: {probs}")
    print(f"Predicted index: {pred_index}, species: {species}, confidence: {confidence}")

    with engine.begin() as conn:
        conn.execute(predictions_table.insert().values(
            user_id=user_id,
            predicted_class=species,
            confidence=confidence,
            timestamp=datetime.now()
        ))
    return {"predicted_class": species, "confidence": confidence}

# --- YOLO Insect Detection ---
yolo_bee = YOLO("bee_best.pt")
yolo_butterfly = YOLO("butterfly_best.pt")
def save_temp(file):
    path = "temp.jpg"
    with open(path, "wb") as f: f.write(file.read())
    return path

@app.post("/predict_insect/")
async def predict_insect(user_id: int = Form(...), file: UploadFile = File(...)):
    path = save_temp(file.file)
    bee_result = yolo_bee(path)[0]
    butterfly_result = yolo_butterfly(path)[0]
    bee_conf = bee_result.boxes.conf[0].item() if bee_result.boxes else 0
    butterfly_conf = butterfly_result.boxes.conf[0].item() if butterfly_result.boxes else 0
    if bee_conf == 0 and butterfly_conf == 0:
        os.remove(path)
        raise HTTPException(status_code=404, detail="No insect detected.")
    if bee_conf > butterfly_conf:
        pred = bee_result.names[bee_result.boxes.cls[0].item()]
        conf = bee_conf
    else:
        pred = butterfly_result.names[butterfly_result.boxes.cls[0].item()]
        conf = butterfly_conf
    os.remove(path)
    with engine.begin() as conn:
        conn.execute(predictions_table.insert().values(
            user_id=user_id,
            predicted_class=pred,
            confidence=conf,
            timestamp=datetime.now()
        ))
    return {"predicted_class": pred, "confidence": conf}

# --- Auth ---
@app.post("/signup/")
async def signup(username: str = Form(...), password: str = Form(...)):
    with engine.begin() as conn:
        if conn.execute(select(users_table).where(users_table.c.username == username)).first():
            raise HTTPException(status_code=400, detail="Username exists.")
        conn.execute(users_table.insert().values(username=username, password=password))
    return {"message": "Signup successful"}

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    with engine.connect() as conn:
        user = conn.execute(select(users_table).where(users_table.c.username == username)).first()
        if user and user.password == password:
            return {"user_id": user.id}
        raise HTTPException(status_code=401, detail="Invalid credentials")

# --- Admin APIs ---
@app.get("/admin/total_users/")
async def total_users():
    with engine.connect() as conn:
        return {"total_users": conn.execute(func.count(users_table.c.id)).scalar()}

@app.get("/admin/total_predictions/")
async def total_predictions():
    with engine.connect() as conn:
        return {"total_predictions": conn.execute(func.count(predictions_table.c.id)).scalar()}

@app.get("/admin/all_predictions/")
async def all_predictions():
    with engine.connect() as conn:
        rows = conn.execute(predictions_table.select().order_by(desc(predictions_table.c.timestamp))).fetchall()
        return [
            {
                "user_id": row.user_id,
                "predicted_class": row.predicted_class,
                "confidence": row.confidence,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None
            }
            for row in rows
        ]

# --- User History ---
@app.get("/user/history/")
async def user_history(user_id: int):
    with engine.connect() as conn:
        rows = conn.execute(
            select(predictions_table.c.predicted_class, predictions_table.c.confidence, predictions_table.c.timestamp)
            .where(predictions_table.c.user_id == user_id)
            .order_by(desc(predictions_table.c.timestamp))
            .limit(100)
        ).fetchall()
        return {"history": [
            {
                "Class": row.predicted_class,
                "Confidence (%)": round(row.confidence * 100, 2),
                "Time": row.timestamp.isoformat() if row.timestamp else ""
            }
            for row in rows
        ]}

# --- Delete user history (Admin only) ---
@app.post("/admin/delete_user_history/")
async def delete_user_history(user_id: int = Form(...)):
    with engine.begin() as conn:
        conn.execute(
            delete(predictions_table).where(predictions_table.c.user_id == user_id)
        )
    return {"message": "Deleted."}

@app.get("/")
async def root():
    return {"message": "Animal detection API with T5 and distilgpt2 educational article"}
