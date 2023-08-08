import sys
import os
import subprocess

command = "python -m spacy download pl_core_news_sm"
subprocess.run(command, shell=True)


from src.model_deployment.utils.nlp_utils import process_tokenize_input, get_ort_session, run_inference
import streamlit as st
from typing import Tuple



# Set page configuration
st.set_page_config(
    page_title="Polish Hate-Speech Detection",
    page_icon="🚫",
    layout="wide"
)


# Function to classify text
def classify_text(input_text: str) -> Tuple[int, float]:
    ort_session = get_ort_session()
    ort_inputs = process_tokenize_input([input_text, ], preprocess=True)
    output_classes, confidences, _ = run_inference(ort_session, ort_inputs)
    return output_classes[0], round(confidences[0], 2)


def main():
    # Set app title and description
    st.title("Polish Hate-Speech Detection Model")
    st.write("This app classifies text as hate speech or not.")

    # Add repository link
    st.sidebar.write("[Repository](https://dagshub.com/a-s-gorski/polish-hatespeech-sentiment-classification)")

    # Add input text box
    input_text = st.text_area("Input Text:", "")

    if st.button("Classify"):
        if input_text:
            predicted_class, confidence = classify_text(input_text)
            if predicted_class == 1:
                st.error("Hate Speech Detected!")
            else:
                st.success("No Hate Speech Detected.")
            st.info(f"Model Confidence: {confidence}%")
        else:
            st.warning("Please enter some text to classify.")


if __name__ == "__main__":
    main()
