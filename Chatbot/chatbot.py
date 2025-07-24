import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import wikipediaapi

# --- Page Setup ---
st.set_page_config(page_title="Animal QA Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🦓 Animal Species QA Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask anything about an animal species below 👇")

# --- Load Model and Tokenizer ---
model_path = "species-t5-finetuned-1"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Wikipedia Setup ---
wiki = wikipediaapi.Wikipedia(language="en", user_agent="SpeciesQA/1.0")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "species" not in st.session_state:
    st.session_state.species = ""

# --- Chat Bubble Display ---
def chat_bubble(msg, is_user=True):
    align = "flex-start" if is_user else "flex-end"
    color = "#dcf8c6" if is_user else "#f1f0f0"
    bubble = f"""
    <div style='display: flex; justify-content: {align}; margin: 8px 0;'>
        <div style='max-width: 70%; background-color: {color}; padding: 10px 15px;
                    border-radius: 15px; font-size: 16px; color: black;'>
            {msg}
        </div>
    </div>
    """
    st.markdown(bubble, unsafe_allow_html=True)

# --- Section Detection ---
def detect_section_from_question(question):
    q = question.lower()
    if any(x in q for x in ["eat", "diet", "food"]): return "Diet"
    if any(x in q for x in ["live", "habitat"]): return "Habitat"
    if any(x in q for x in ["behavior", "behaviour", "social"]): return "Behaviour"
    if any(x in q for x in ["reproduce", "reproduction"]): return "Reproduction"
    if any(x in q for x in ["threat", "danger", "extinct"]): return "Threats"
    return ""

# --- Context Fetching ---
def get_context(species, question):
    section = detect_section_from_question(question)
    page = wiki.page(species)
    if not page.exists():
        return "No Wikipedia content found for that species."
    for s in page.sections:
        if section and s.title.lower() == section.lower():
            return s.text
    return page.summary

# --- Model Answer Generation ---
def generate_answer(question, species):
    context = get_context(species, question)
    input_text = f"question: {question}  context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- Top Input: Species ---
species_input = st.text_input("🔍 Enter the species name (e.g., Panthera leo)", value=st.session_state.species)
if species_input:
    st.session_state.species = species_input

# --- Main Input: Question ---
question = st.text_input("💬 Ask your question:")

if st.button("Ask"):
    if question and st.session_state.species:
        answer = generate_answer(question, st.session_state.species)
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("bot", answer))
        st.rerun()
    elif not st.session_state.species:
        st.warning("Please enter a species name first.")

# --- Chat History Display ---
st.markdown("### Chat History")
for speaker, msg in st.session_state.chat_history:
    chat_bubble(msg, is_user=(speaker == "user"))

# --- Clear Chat ---
if st.button("🔁 Start Over"):
    st.session_state.chat_history = []
    st.session_state.species = ""
    st.rerun()
