# NOTE

Dataset used is from "WASSA 2017 Shared Task on Emotion Intensity"

Steps:
1. Run Dataset_splitter.py first to split data into test,training and validation datasets.
2. Train the models using "Train.py"
3. Connect the LM Studio Server API in "Run.py" for responses.

In code comments are present for better understanding

# 🌌 Emotional-AI

An **Emotion-Aware Interaction (EAI)** system that combines emotion detection with language model–based dialogue generation to simulate emotionally intelligent conversations.  

The project focuses on detecting and interpreting four primary emotions from user text:  
- Anger  
- Fear  
- Joy  
- Sadness  

These detected emotional intensities are then used to guide the response of a conversational agent, making interactions more empathetic and context-aware.  

---

## 🚀 Features
- Independent emotion regression models built with **DistilBERT**.  
- Emotion intensities normalized and categorized into ranges (e.g., mild, moderate, extreme).  
- **Persona logic** that evolves emotional tone across conversations.  
- Local integration with **LM Studio** for generating context-sensitive responses.  
- Simple **Tkinter-based chat interface** for interactive testing.  

---

## 📊 Example
**Input:**  
> "I am very sad today."  

**Predicted Intensities:**  
- Sadness: 0.85  
- Joy: 0.05  
- Anger: 0.10  
- Fear: 0.10  

**Categorization:**  
- Extremely sad, not joyful, slightly angry, slightly fearful  

**Generated Response:**  
> "I’m really sorry you’re feeling this way. I’m here for you—want to talk about what happened?"  

---

## 📌 Roadmap
- Upload core code and pretrained models.  
- Add GUI integration with evolving persona emotions.  
- Expand dataset coverage for sarcasm and multi-emotion handling.  
- Release evaluation metrics and survey-based analysis.  

---

## 🛠️ Status
This repository is a **work in progress**. Code and models will be uploaded in pieces. For now, this README provides an overview of the project goals and design.  