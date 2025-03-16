import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
import os

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api

# تحميل مكتبة Punkt إذا لم تكن محملة مسبقًا
nltk.data.path.append("./nltk_data")  # تحديد مجلد تخزين
if not os.path.exists("./nltk_data/tokenizers/punkt"):
    nltk.download('punkt', download_dir="./nltk_data")

# تحميل نموذج Word2Vec
@st.cache_resource
def load_w2v():
    return api.load("word2vec-google-news-300")

w2v_model = load_w2v()

# إعدادات النموذج
embedding_dim = 300  # حجم المتجهات
max_length = 30  # الحد الأقصى لطول الجملة

# دالة لتحويل النص إلى قائمة من المتجهات
def text_to_sequence(text, model):
    words = word_tokenize(text.lower())
    sequence = []
    for word in words:
        if word in model:
            sequence.append(model[word])
        else:
            sequence.append(np.zeros(embedding_dim))  # استبدال الكلمات غير الموجودة بصفوف من الأصفار
    return sequence

# تحميل نموذج CNN
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./cnn_model_headline.h5")

model = load_model()

# بناء واجهة Streamlit
st.title("News Headline Classification App")
st.write("Enter a news headline to classify it using the trained CNN model.")

headline = st.text_input("Enter a headline:")

if st.button("Predict"):
    if headline:
        # تحويل النص إلى متجهات
        sequence = text_to_sequence(headline, w2v_model)
        padded_sequence = pad_sequences([sequence], maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0)
        
        # توقع التصنيف
        prediction = model.predict(padded_sequence)[0][0]
        label = "Positive" if prediction > 0.7 else "Negative"
        
        st.write(f"**Prediction:** {label} (Confidence: {prediction:.4f})")
    else:
        st.write("⚠️ Please enter a headline.")
