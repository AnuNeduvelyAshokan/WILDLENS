# Species Information Retrieval - T5 Fine-Tuning with Custom QA Dataset (with Dynamic Wikipedia Retrieval)

#  STEP 1: Install Required Libraries
!pip install transformers datasets evaluate rouge_score requests

# Disable W&B tracking to avoid API key prompts
import os
os.environ["WANDB_DISABLED"] = "true"

# STEP 2: Load Dataset
import pandas as pd
from datasets import Dataset

df = pd.read_csv("Generated_QA_Dataset_Updated.csv")  # Update path if needed
dataset = Dataset.from_pandas(df)
train_test = dataset.train_test_split(test_size=0.2)

#  STEP 3: Preprocess & Tokenize
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")

def preprocess(example):
    input_text = f"question: {example['question']}  context: {example['context']}"
    target_text = example["answer"]
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_text, max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = train_test.map(preprocess)

#  STEP 4: Load and Train the Model
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

model = T5ForConditionalGeneration.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer.model_input_names = ["input_ids", "attention_mask", "labels"]

training_args = TrainingArguments(
    output_dir="./species-t5-finetuned",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()

#  STEP 5: Evaluate
from collections import Counter
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Custom token-level F1
def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate_model(dataset):
    predictions, references, f1_scores = [], [], []
    model.eval()
    for item in dataset:
        input_text = f"question: {item['question']}  context: {item['context']}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output = model.generate(input_ids.to(device), max_length=64)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        ref = item["answer"]
        predictions.append(pred)
        references.append(ref)
        f1_scores.append(compute_f1(pred, ref))
    print("BLEU:", bleu.compute(predictions=predictions, references=[[r] for r in references]))
    print("ROUGE:", rouge.compute(predictions=predictions, references=references))
    print("F1 (avg):", sum(f1_scores) / len(f1_scores))

# Run evaluation
evaluate_model(train_test["test"])

!pip install wikipedia-api

#  STEP 6: Dynamic Wikipedia Context Retrieval
import wikipediaapi

# Set custom user agent per Wikipedia policy
wiki = wikipediaapi.Wikipedia(language="en", user_agent="SpeciesQA/1.0 (your_email@example.com)")

def detect_section_from_question(question):
    question = question.lower()

    if any(q in question for q in ["eat", "diet", "food"]):
        return "Diet"
    elif any(q in question for q in ["live", "habitat", "location", "found", "range"]):
        return "Habitat"
    elif any(q in question for q in ["behavior", "behaviour", "social", "act"]):
        return "Behaviour"
    elif any(q in question for q in ["reproduce", "reproduction", "birth", "offspring"]):
        return "Reproduction"
    elif any(q in question for q in ["threat", "danger", "extinct", "predator", "vulnerable"]):
        return "Threats"
    elif any(q in question for q in ["kind of species", "what species", "type of animal", "category"]):
        return "Summary"
    elif any(q in question for q in ["where does it", "locate", "found", "native to"]):
        return "Habitat"
    elif any(q in question for q in ["stable", "population", "conservation status"]):
        return "Conservation"
    elif any(q in question for q in ["rare", "scarce", "uncommon"]):
        return "Conservation"
    else:
        return ""

def get_species_context(species_name, question):
    section = detect_section_from_question(question)
    page = wiki.page(species_name)
    if not page.exists():
        return "No Wikipedia content found."
    for s in page.sections:
        if section and s.title.lower() == section.lower():
            return s.text
    return page.summary  # fallback

def generate_dynamic_answer(question, species_name):
    context = get_species_context(species_name, question)
    print("\n📘 Context:", context)
    input_text = f"question: {question}  context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

model.save_pretrained("species-t5-finetuned")
tokenizer.save_pretrained("species-t5-finetuned")

#  STEP 7: Save Model (Optional)
model.save_pretrained("species-t5-finetuned")
tokenizer.save_pretrained("species-t5-finetuned")

# 🙋‍♂️ STEP 8: User Input Interface
print("\n🔍 Ask the model a species-related question!")
user_question = input("Enter your question: ")
user_species = input("Enter the species name (e.g., Panthera leo): ")

user_answer = generate_dynamic_answer(user_question, user_species)
print("\n🤖 Answer:", user_answer)

import shutil
shutil.make_archive("species-t5-finetuned", 'zip', "species-t5-finetuned")