# Wild Lens: Wildlife Recognition and Education Assistant

Wild Lens is an AI-powered web application that combines **Computer Vision** and **Natural Language Processing** to identify wildlife species from images and provide **contextual educational content** in real time.

>  Works locally via Streamlit — see instructions below

---

## Features

- **YOLOv8 Object Detection** for identifying bees and butterflies  
- **Fast R-CNN Detection** for larger wild animals like lions, cheetahs, wolves, etc.  
- **Hugging Face NLP Modules** for educational text generation  
- **Fine-tuned T5 Model** creates readable species descriptions  
- **PostgreSQL Integration** to log predictions and stats  
- **Modular Streamlit App** with multi-model selection and session tracking  
- **User-friendly UI** with real-time visual feedback and educational summaries

---

## Models Used

| Task                    | Model                     | Framework    |
|-------------------------|---------------------------|--------------|
| Bee Detection           | YOLOv8n                   | Ultralytics  |
| Butterfly Detection     | YOLOv8s                   | Ultralytics  |
| Animal Detection        | Fast R-CNN (`.pth` model) | PyTorch      |
| Animal Classification   | SimpleCNN (`.pth`)        | PyTorch      |
| Species Info Retrieval  | Custom NLP Classifier     | Hugging Face |
| Educational Text Gen.   | Fine-tuned T5 (`checkpoint-360`) | Hugging Face |

---

## Core Project Structure

wild-lens/
|-- app.py
|-- species_data.py
|-- __init__.py
|-- datasets
  |-- Generated_QA_Dataset.csv
|-- pages
  |-- detection_analytics.py
  |-- performance_metrics.py
|--
  |-- species_data.cpython-311.pyc
  |-- __init__.cpython-311.pyc
|-- utils
  |-- chatbot_core.py
  |-- database.py
  |-- detection.py
  |-- fastrcnn_model.py
  |-- info_retreival_model_final.ipynb
  |-- offline_mode.py
  |-- species-t5-finetuned.zip
  |-- species_locations.py
  |-- species_map.py
  |-- summarizer.py
  |-- t5_model.py
  |-- text_summarization.py
  |-- text_summarization_model.ipynb
  |-- visualization.py
  |--
    |-- chatbot_core.cpython-311.pyc
    |-- database.cpython-311.pyc
    |-- detection.cpython-311.pyc
    |-- offline_mode.cpython-311.pyc
    |-- species_map.cpython-311.pyc
    |-- visualization.cpython-311.pyc
|-- models
  |-- bee_best.pt
  |-- butterfly_best.pt
  |-- fast_rcnn
    |-- animal_classifier.pth
    |-- fast_rcnn.py
    |--
      |-- fast_rcnn.cpython-311.pyc
  |-- hugging_face_t5
    |-- species-t5-finetuned
      |-- checkpoint-360
        |-- added_tokens.json
        |-- config.json
        |-- generation_config.json
        |-- model.safetensors
        |-- optimizer.pt
        |-- rng_state.pth
        |-- scheduler.pt
        |-- special_tokens_map.json
        |-- spiece.model
        |-- tokenizer_config.json
        |-- trainer_state.json
        |-- training_args.bin


---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (optional but recommended)
- PostgreSQL (v16 or 17)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

### Installation

```bash
git clone https://github.com/yourusername/wild-lens.git
cd wild-lens
pip install -r requirements.txt
streamlit run app.py
```
## Example Workflow
- Upload an image of a wildlife species (e.g., bee, butterfly, lion)
- The selected model (YOLO or Fast R-CNN) detects the species
- The NLP classifier identifies the species context
- The T5 model generates an educational description
- All results are shown in the app and logged in PostgreSQL

## Research Basis
- Wild Lens integrates state-of-the-art models:
- YOLOv8 for real-time object detection
- Fast R-CNN for high-accuracy classification
- Hugging Face T5 for context-aware natural language generation

It promotes biodiversity education, supports citizen science, and demonstrates the power of AI for ecological impact.

## Future Work
- Add more species (birds, plants, amphibians)
- Multilingual support (Spanish, Arabic, etc.)
- Offline mobile app deployment

## Contributors
- Hassan Riaz – Lead Developer, YOLO/Streamlit integration
- Jennifer Henry Dominique – NLP Classifier & Species Knowledge Base
- Anu Neduvely Ashokan – T5 Fine-tuning & Language Generation
- Khurrum Rehman – Fast R-CNN Development and Testing

# Acknowledgments
- Ultralytics for YOLOv8
- Hugging Face Transformers
- Fast R-CNN (PyTorch Tutorials)
- iNaturalist Dataset & Kaggle Contributors

