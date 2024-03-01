import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# Streamlit UI
st.title('SDG&E APP')

############### IMAGE ANALYSIS ###############

# Image upload UI
st.write("Upload an image of a utility pole for analysis:")
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Assuming we have a function to process the image and analyze it:
# process_uploaded_image(uploaded_image)

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    
    # If you there is an image analysis model ready, process the image here
    # and display the results. For example:
    # result = process_uploaded_image(uploaded_image)
    # st.write(f"Image analysis result: **{result}**")

# Note: we'll need to implement the process_uploaded_image function based on our model.


############### TEXT ANALYSIS ###############

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()
sia.lexicon.update({"collision": -4.0, "collide": -4.0, "collided": -4.0, "broke": -3.0, "broken": -3.0, \
                           "damaged": -2.0, "damage": -2.0, "corrosion": -2.0, "rust": -2.0, "storm": -3.5, \
                           "crash": -4.0, "lean": -2.0, "unstable": -2.0, "low-hanging": -1.5, "wire": -1.0, \
                           "outage": -4, "expose": -2, "fire": -4.0, "spark": -3.0, "smoke": -3.0, \
                           "flame": -4.0, "overgrown": -1.5, "tree": -1.0, "noise": -1.0, "sound": -1.0})
# with open('../models/svm_model.pkl', 'rb') as f:
#     sgd = pickle.load(f)

# Text analysis UI
user_input = st.text_area("Is there something you'd want us to know?")

# Add an "Analyze" button for text sentiment analysis
if st.button('Submit'):
    # Perform sentiment analysis only if the button is pressed
    scores = sia.polarity_scores(user_input)
    #topic = sgd.predict([user_input])
    sentiment = 'Neutral'
    if scores['compound'] > 0.05:
        sentiment = 'Positive'
    elif scores['compound'] < -0.05:
        sentiment = 'Negative'
    
    # Display the sentiment result
    st.write(f"Sentiment: **{sentiment, scores['compound']}**")
    #st.write(f"Sentiment: **{topic}**")