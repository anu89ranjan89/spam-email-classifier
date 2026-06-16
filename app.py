import streamlit as st
import pickle

# Page Config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Load model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Header
st.title("📧 Spam Email Classifier")
st.markdown(
    """
    Detect whether a message is **Spam** or **Ham (Not Spam)** using
    Machine Learning, TF-IDF, and Naive Bayes.
    """
)

st.divider()

# Input
message = st.text_area(
    "Enter your message",
    height=150,
    placeholder="Type or paste your email/message here..."
)

if st.button("🔍 Analyze Message", use_container_width=True):

    if message.strip():

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)[0]

        probabilities = model.predict_proba(transformed)

        spam_prob = probabilities[0][1]

        ham_prob = probabilities[0][0]

        st.divider()

        if prediction == "spam":
            st.error("⚠️ Spam Message Detected")
        else:
            st.success("✅ Legitimate Message (Ham)")

        st.subheader("Prediction Confidence")

        st.progress(float(spam_prob))

        st.write(f"📛 Spam Probability: **{spam_prob*100:.2f}%**")
        st.write(f"📨 Ham Probability: **{ham_prob*100:.2f}%**")

    else:
        st.warning("Please enter a message.")