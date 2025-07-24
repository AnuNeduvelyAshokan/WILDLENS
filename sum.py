import streamlit as st
import wikipedia
from transformers import pipeline

# --- Streamlit Page Settings ---
st.set_page_config(page_title="Animal Species Info Generator", layout="centered")
st.title("🦁 Animal Species & Class Information")


st.sidebar.header("Settings")
max_length = st.sidebar.slider("Max Output Length", min_value=50, max_value=300, value=150)
temperature = st.sidebar.slider("Creativity Level (Temperature)", min_value=0.0, max_value=1.0, value=0.3)


@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-xl")

generator = load_model()


animal_input = st.text_input(
    "🔍 Enter an animal name or species (e.g., 'Bengal tiger', 'Large White butterfly', 'Great Horned Owl'):",
    ""
)


def build_prompt(animal):
    return (
        f"Provide accurate scientific information about the animal species '{animal}'. "
        f"Include its biological class (e.g., Mammalia), habitat, and 2–3 key physical or behavioral traits. "
        f"Ensure the class and details are biologically accurate."
    )


def get_wikipedia_summary(query):
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
        return summary
    except Exception:
        return None


if st.button("Generate Information") and animal_input.strip():
    with st.spinner("Gathering factual content..."):
        # Try Wikipedia first
        wiki_result = get_wikipedia_summary(animal_input.strip())

        if wiki_result:
            st.subheader("📘 Wikipedia Summary:")
            st.write(wiki_result)
        else:
            
            prompt = build_prompt(animal_input.strip())
            output = generator(prompt, max_length=max_length, temperature=temperature)[0]["generated_text"]
            st.subheader("🤖 Model-Generated Content:")
            st.write(output)


st.markdown("---")
st.markdown("✅ Educational app using Hugging Face 🤗 + Streamlit 🎈 + Wikipedia 🌐")

