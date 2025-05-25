import spacy
import streamlit as st

# Load trained spaCy model
nlp = spacy.load("output/model-best")  # adjust path if needed

st.title("GoEmotions Predictor")
st.write("This version simulates top 5 predicted emotions for any input. Great for testing your UI!")

# Text input
text = st.text_area("Type your message here:")

if st.button("Show Top Emotions"):
    if text.strip():
        doc = nlp(text)
        cats = doc.cats
        # Get top 5 predictions
        top_emotions = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:5]
        st.subheader("Top 5 Emotions:")
        for label, score in top_emotions:
            st.write(f"**{label}**: {score:.3f}")
    else:
        st.warning("Please enter some text.")
