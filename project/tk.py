import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

model_filename = "finalized_model.pkl"
vectorizer_filename = "vectorizer_model.pkl"
pac = pickle.load(open(model_filename, "rb"))
vector = pickle.load(open(vectorizer_filename, "rb"))

def classify_text():
    text = text_box.get("1.0", "end-1c")
    if not text.strip():
        messagebox.showwarning("Input Error", "Please enter some text to classify.")
        return
    
    text_transformed = vector.transform([text])
    prediction = pac.predict(text_transformed)[0]

    result_label.config(text=f"Prediction: {prediction}")

window = tk.Tk()
window.title("Fake News Classifier")

title_label = tk.Label(window, text="Enter News Text to Classify", font=("Helvetica", 16))
title_label.pack(pady=10)

text_box = tk.Text(window, height=10, width=50)
text_box.pack(pady=10)

classify_button = tk.Button(window, text="Classify", font=("Helvetica", 14), command=classify_text)
classify_button.pack(pady=10)

result_label = tk.Label(window, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=10)

window.mainloop()
