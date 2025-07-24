#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from transformers import pipeline
import torch


# In[2]:


def load_gpt_pipeline(model_name="EleutherAI/gpt-neo-2.7B"):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using {'GPU' if device == 0 else 'CPU'} for inference.")
    return pipeline("text-generation", model=model_name, device=device)


# In[3]:


model_name = "EleutherAI/gpt-neo-2.7B"
text_gen_pipeline = load_gpt_pipeline(model_name)


# In[4]:


def get_wikipedia_data(species_name):
    print("Fetching information from Wikipedia...")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{species_name}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get("extract", "No information available.")
        return "No data found on Wikipedia."
    except requests.exceptions.RequestException as error:
        return f"Error fetching data: {error}"


# In[5]:


def create_educational_content(pipeline, species_name, facts, max_length=128):
    facts = facts[:500]
    prompt = (f"Write an educational article about the species {species_name}. "
              f"Include its habitat, threats, conservation status, and interesting facts. "
              f"Details: {facts}")
    try:
        print("Generating educational content...")
        result = pipeline(
            prompt,
            max_new_tokens=max_length,
            truncation=True,
            num_return_sequences=1,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        return result[0]["generated_text"]
    except Exception as error:
        return f"Error generating content: {error}"


# In[6]:


print("---------------Welcome to the Wildlife Education Generator-----------------")
species_name = input("Enter the name of the species you'd like to learn about: ").strip()
wiki_data = get_wikipedia_data(species_name)
print("\nWikipedia Summary:")
print(wiki_data)
educational_content = create_educational_content(text_gen_pipeline, species_name, wiki_data)
print("\nGenerated Educational Content:")
print(educational_content)


# 
