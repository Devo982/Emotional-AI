import requests
import json
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import tkinter as tk
from tkinter import scrolledtext

# Step 1: Define the BERTForRegression class
class BERTForRegression(nn.Module):
    def __init__(self, bert_model_name='distilbert-base-uncased'):
        super(BERTForRegression, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.bert_output_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.bert_output_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        regression_output = self.sigmoid(self.regressor(cls_output))
        return regression_output

# Step 2: Load the trained DistilBERT models and predict intensities
def predict_intensity(model, tweet, device='cuda'):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()
    model.to(device)

    encoding = tokenizer(
        tweet,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        intensity = output.item()
    return intensity

def load_and_predict_emotions(tweet, device='cpu'):
    emotions = {}
    model_paths = {
        'anger': 'models/bert_regression_anger.pt',
        'fear': 'models/bert_regression_fear.pt',
        'joy': 'models/bert_regression_joy.pt',
        'sadness': 'models/bert_regression_sadness.pt'
    }

    for emotion, model_path in model_paths.items():
        try:
            model = BERTForRegression()
            model.load_state_dict(torch.load(model_path, map_location=device))
            intensity = predict_intensity(model, tweet, device)
            emotions[emotion] = intensity
        except Exception as e:
            print(f"Error loading model for {emotion}: {str(e)}")
            emotions[emotion] = {
                "anger": 0.3,
                "fear": 0.3,
                "joy": 0.3,
                "sadness": 0.3
            }[emotion]
    return emotions

# Step 3: Categorize emotion intensities with ranges adjusted for data bias
def categorize_emotions(emotions):
    ranges = {
        "anger": [
            (0.0, 0.4, "not angry"),
            (0.4, 0.5, "mildly angry"),
            (0.5, 0.7, "moderately angry"),
            (0.7, 1.0, "extremely angry")
        ],
        "fear": [
            (0.0, 0.3, "not fearful"),
            (0.3, 0.5, "mildly fearful"),
            (0.5, 0.7, "moderately fearful"),
            (0.7, 1.0, "extremely fearful")
        ],
        "joy": [
            (0.0, 0.2, "not joyful"),
            (0.2, 0.4, "mildly joyful"),
            (0.4, 0.6, "moderately joyful"),
            (0.6, 1.0, "extremely joyful")
        ],
        "sadness": [
            (0.0, 0.3, "not sad"),
            (0.3, 0.5, "mildly sad"),
            (0.5, 0.7, "moderately sad"),
            (0.7, 1.0, "extremely sad")
        ]
    }

    categorized = {}
    for emotion, intensity in emotions.items():
        for lower, upper, label in ranges[emotion]:
            if lower <= intensity < upper:
                categorized[emotion] = label
                break
        if emotion not in categorized:
            categorized[emotion] = ranges[emotion][-1][2]

    return categorized

# Step 4: Generate emotionally appropriate response using LM Studio's LLaMA API
def generate_response(categorized_emotions, tweet, emotions, persona_emotions, api_url="http://26.219.57.155:1234/v1/chat/completions"):
    dominant_emotion = max(emotions, key=emotions.get)
    secondary_emotion = sorted(emotions, key=emotions.get, reverse=True)[1] 
    emotion_summary = ", ".join([f"{label}" for label in categorized_emotions.values()])

    # Construct persona's emotional tone based on the examples
    tone_description = []
    if persona_emotions['anger'] == 'extremely':
        tone_description.append("extremely angry")
    elif persona_emotions['anger'] == 'moderate':
        tone_description.append("moderately angry")
    elif persona_emotions['anger'] == 'slight':
        tone_description.append("slightly angry")

    if persona_emotions['joy'] == 'slight':
        tone_description.append("slightly happy")
    elif persona_emotions['joy'] == 'moderate':
        tone_description.append("moderately happy")
    elif persona_emotions['joy'] == 'extremely':
        tone_description.append("extremely happy")

    if persona_emotions['fear'] == 'slight':
        tone_description.append("slightly scared")
    elif persona_emotions['fear'] == 'moderate':
        tone_description.append("moderately scared")
    elif persona_emotions['fear'] == 'extremely':
        tone_description.append("extremely scared")

    if persona_emotions['sadness'] == 'slight':
        tone_description.append("slightly sad")
    elif persona_emotions['sadness'] == 'moderate':
        tone_description.append("moderately sad")
    elif persona_emotions['sadness'] == 'extremely':
        tone_description.append("extremely sad")

    tone = ", ".join(tone_description) if tone_description else "neutral"

    system_message = (
    f"You are the person's friend with a {tone} tone and NOT an AI. Respond to the user in a reactive way, and respond in a similar manner to what the user does"
    )
    user_message = (
        f"The user said '{tweet}'.\n"
        f"Their emotions are {emotion_summary}.\n"
        f"They are primarily feeling {categorized_emotions[dominant_emotion]} "
        f"and secondarily {categorized_emotions[secondary_emotion]}.\n"
        f"React in a mirrored tone and strictly no sarcasm, and do not address these settings in your replies, keep it concise, and act human and be curious"
    )

    payload = {
        "model": "llama-3.2-1b-instruct",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        generated_text = result["choices"][0]["message"]["content"]
        return generated_text.strip()
    except requests.RequestException as e:
        return f"Error calling LM Studio API: {str(e)}"

# Step 5: Process the tweet and return results
def process_tweet(tweet, persona_emotions):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotions = load_and_predict_emotions(tweet, device)
    
    intensity_output = "Recalculated Emotion Intensities:\n"
    for emotion, intensity in emotions.items():
        intensity_output += f"{emotion.capitalize()}: {intensity:.4f}\n"
    
    categorized_emotions = categorize_emotions(emotions)
    
    categorized_output = "Categorized Emotions:\n"
    for emotion, label in categorized_emotions.items():
        categorized_output += f"{emotion.capitalize()}: {label}\n"
    
    response = generate_response(categorized_emotions, tweet, emotions, persona_emotions)
    
    return intensity_output, categorized_output, response

# Step 6: Create the GUI with emotional tone adjustment
class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Analyzer Chat")
        
        # Persona emotional state (evolves based on conversation)
        self.persona_emotions = {
            'anger': 'none',
            'joy': 'mildly',
            'fear': 'none',
            'sadness': 'none'
        }
        self.conversation_step = 0  # Track the conversation progress to adjust emotions
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(root, width=50, height=20, wrap=tk.WORD,font=("Helvetica",14))
        self.chat_display.pack(padx=10, pady=10)
        self.chat_display.config(state='disabled')
        
        # Input area
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.input_field = tk.Entry(self.input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_field.bind("<Return>", self.send_message)
        
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
    
    def update_persona_emotions(self, user_emotions):
        self.persona_emotions = {}

        for emotion, intensity in user_emotions.items():
            if intensity >= 0.75:
                self.persona_emotions[emotion] = 'extremely'
            elif intensity >= 0.5:
                self.persona_emotions[emotion] = 'moderate'
            elif intensity >= 0.3:
                self.persona_emotions[emotion] = 'slight'
            else:
                self.persona_emotions[emotion] = 'none'
    
    def send_message(self, event=None):
        tweet = self.input_field.get().strip()
        if not tweet:
            return

        user_emotions = load_and_predict_emotions(tweet, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.update_persona_emotions(user_emotions)
        
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Display user message
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"You: {tweet}\n\n")
        
        # Process the tweet and get results
        intensity_output, categorized_output, response = process_tweet(tweet, self.persona_emotions)
        
        # Display results
        self.chat_display.insert(tk.END, intensity_output + "\n")
        self.chat_display.insert(tk.END, categorized_output + "\n")
        self.chat_display.insert(tk.END, f"Response: {response}\n\n")
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatGUI(root)
    root.mainloop()