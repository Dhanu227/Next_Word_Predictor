import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import shap

# Load the LSTM Model
model = load_model('Models/next_word_lstm.h5')

# Load the tokenizer
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict top N next words
def predict_top_n_words(model, tokenizer, text, max_sequence_len, top_n=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = predicted_probs.argsort()[-top_n:][::-1]
    
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    top_words = [(index_word.get(i, "UNK"), predicted_probs[i]) for i in top_indices]
    
    return top_words

# Function for token contribution
def token_contribution_analysis(model, tokenizer, text, max_sequence_len, target_word):
    words = text.split()
    original_tokens = tokenizer.texts_to_sequences([text])[0]
    original_seq = pad_sequences([original_tokens[-(max_sequence_len-1):]], maxlen=max_sequence_len-1, padding='pre')
    original_probs = model.predict(original_seq, verbose=0)[0]
    target_index = tokenizer.word_index.get(target_word, None)
    
    if target_index is None:
        return None

    base_prob = original_probs[target_index]
    importance_scores = []

    for i in range(len(words)):
        modified = words[:i] + ["UNK"] + words[i+1:]
        modified_tokens = tokenizer.texts_to_sequences([" ".join(modified)])[0]
        modified_seq = pad_sequences([modified_tokens[-(max_sequence_len-1):]], maxlen=max_sequence_len-1, padding='pre')
        modified_probs = model.predict(modified_seq, verbose=0)[0]
        drop = base_prob - modified_probs[target_index]
        importance_scores.append((words[i], drop))

    return sorted(importance_scores, key=lambda x: x[1], reverse=True)


# ✅ INPUT TEXT
input_text = st.text_input("Enter a sequence of words", "To be or not to")

# Run prediction and show output
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    top_words = predict_top_n_words(model, tokenizer, input_text, max_sequence_len)

    st.write("### 🔮 Top Predictions")
    for word, prob in top_words:
        st.write(f"- **{word}**: {prob:.4f}")

    df = pd.DataFrame(top_words, columns=["Word", "Probability"])
    st.bar_chart(df.set_index("Word"))

    most_probable_word = top_words[0][0]
    contributions = token_contribution_analysis(model, tokenizer, input_text, max_sequence_len, most_probable_word)

    if contributions:
        st.write(f"### 🧠 Token Importance (for: *{most_probable_word}*)")
        df_contrib = pd.DataFrame(contributions, columns=["Token", "Importance"])
        st.bar_chart(df_contrib.set_index("Token"))
