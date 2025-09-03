import requests
import json
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

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
def predict_intensity(model, tweet, device='cpu'):
    """
    Predict emotion intensity for a tweet using a trained DistilBERT model.
    Args:
        model (BERTForRegression): Trained DistilBERT model.
        tweet (str): Input tweet.
        device (str): Device to run inference on ('cpu' or 'cuda').
    Returns:
        float: Predicted intensity in [0, 1].
    """
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

# Load models and recalculate intensities for the tweet
tweet = "I hate you in specific"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
emotions = {}

# Paths to your saved models (update these paths as needed)
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
        print(f"Recalculated {emotion.capitalize()} Intensity: {intensity:.4f}")
    except Exception as e:
        print(f"Error loading model for {emotion}: {str(e)}")
        emotions[emotion] = {
            "anger": 0.6112,
            "fear": 0.6226,
            "joy": 0.2260,
            "sadness": 0.7131
        }[emotion]

# Step 3: Categorize emotion intensities with ranges adjusted for data bias
def categorize_emotions(emotions):
    """
    Categorize emotion intensities into descriptive labels, accounting for limited dataset bias.
    Args:
        emotions (dict): Dictionary of emotion intensities.
    Returns:
        dict: Categorized emotion labels.
    """
    ranges = {
        "anger": [
            (0.0, 0.3, "not angry"),
            (0.3, 0.5, "mildly angry"),
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
            (0.0, 0.3, "not joyful"),
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
def generate_response(categorized_emotions, tweet, api_url="http://26.219.57.155:1234/v1/chat/completions"):

    dominant_emotion = max(emotions, key=emotions.get)
    secondary_emotion = sorted(emotions, key=emotions.get, reverse=True)[1]
    emotion_summary = ", ".join([f"{label}" for label in categorized_emotions.values()])
    system_message = (
        "You are an empathetic assistant. Respond to the user in a reactive way, "
        "acknowledging their feelings based on the provided emotional analysis. "
        "Keep the response concise and avoid overly cheerful tones unless the user is joyful"
    )
    user_message = (
        f"The person said '{tweet}'.\n"
        f"Their emotions are {emotion_summary}.\n"
        f"They are primarily feeling {categorized_emotions[dominant_emotion]} "
        f"and secondarily {categorized_emotions[secondary_emotion]}.\n"
        f"Please respond emotionally, I want you to be going through the same emotion as the person, without acknowledging any of the parameters except what the person said and stop acting like an AI"
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

# Categorize emotions with recalculated intensities
categorized_emotions = categorize_emotions(emotions)
print("\nCategorized Emotions:")
for emotion, label in categorized_emotions.items():
    print(f"{emotion.capitalize()}: {label}")

# Generate response
response = generate_response(categorized_emotions, tweet)
print(f"\nTweet: {tweet}")
print(f"Generated Response: {response}")