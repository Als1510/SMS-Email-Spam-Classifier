from select import select
from click import option
import nltk
import pickle
import string
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
ps = PorterStemmer()

st.set_page_config(page_title='SMS / Email Spam Classifier', layout = 'centered', page_icon = None, initial_sidebar_state = 'expanded')

def local_css(file_name):
  with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

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

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

selected = ''
try_dummies = st.checkbox('Try dummy messages?')
if try_dummies:
  selected = st.selectbox(
    'Select Message',
    ("Congratulations! You've won a $1000 xyz gift card. Go to http://xyz to claim now.", "Your IRS tax refund is pending acceptance. Must accept within 24 hours: http://xyz", "URGENT Your grandson was arrested last night as Mexico. Need bail money immediately Western Unione Wire $9,500 http://xyz","Today am going to college so am not able to attend the class", "Do one thing! Chnage the sentence into: 'Because i want to concentrate in my educational career i'm living here..'", "I walked an hour to see you! Doesn't that show I care about you."))
  
st.title("Email/SMS Spam Classifier")
input = st.text_area("Enter the message", value=selected)
if st.button("Predict"):
  transformed_input = transform_text(input)
  vector_input = tfidf.transform([transformed_input])
  result = model.predict(vector_input)[0]
  if result == 1:
    st.markdown('<h2 class="center warning">Spam</h1>',unsafe_allow_html=True)
  else:
    st.markdown('<h2 class="center success">Not Spam</h1>',unsafe_allow_html=True)