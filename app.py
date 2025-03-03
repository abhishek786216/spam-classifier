import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained models
try:
    tfidf = pickle.load(open('cv.pkl', 'rb'))
    model = pickle.load(open('mb.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'cv.pkl' and 'mb.pkl' are in the correct directory.")

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

# Input box
input_sms = st.text_area("Enter the message", help="Type your email or SMS here and check if it's spam.")

# Predict button
if st.button('Predict 🚀'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display Result
        if result == 1:
            st.error("🚨 **Spam Alert!** This message seems suspicious. Be cautious! 🛑")
        else:
            st.success("✅ **Safe!** This message is not spam. You can trust it. 😊")
