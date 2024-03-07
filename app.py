import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from streamlit_image_select import image_select

# Streamlit UI
st.title('CERA: ML Model Demo')
st.write("The following page presents a rudimentary demonstration of the capabilities of initial pole validation through object detection models, and topic and sentiment identification through NLP (natual language processing) models. In the future, these models can be further developed to identify more visual and written risks to aid both the users and SDG&E.")

############### IMAGE ANALYSIS ###############
    
# Image selection
img = image_select(
    label="Select an example image for analysis",
    images=[
        "data/validation/IMG_6859.jpg",
        "data/validation/IMG_6864.jpg",
        "data/validation/IMG_6875.jpg",
        "data/validation/IMG_6892.jpg",
    ])
# Given the chosen image, display the labelled results
st.write("Image validation results:")
if img == "data/validation/IMG_6859.jpg":
    st.image("models/detr_results/valid02.png")
    st.write("**An electric pole has been identified with 58% confidence.**")
elif img == "data/validation/IMG_6864.jpg":
    st.image("models/detr_results/valid012.png")
    st.write("**An electric pole has been identified with 73% confidence.**")
elif img == "data/validation/IMG_6875.jpg":
    st.image("models/detr_results/valid0122.png")
    st.write("**An electric pole has been identified with 68% confidence.**")
else:
    st.image("models/detr_results/valid0123.png")
    st.write("**An electric pole has been identified with 97% confidence.**")


############### TEXT ANALYSIS ###############

# Initialize and customize the sentiment analyzer
sia = SentimentIntensityAnalyzer()
sia.lexicon.update({"collision": -4.0, "collide": -4.0, "collided": -4.0, "broke": -3.0, "broken": -3.0, \
                           "damaged": -2.0, "damage": -2.0, "corrosion": -2.0, "rust": -2.0, "storm": -3.5, \
                           "crash": -4.0, "lean": -2.0, "unstable": -2.0, "low-hanging": -1.5, "wire": -1.0, \
                           "outage": -4, "expose": -2, "fire": -4.0, "spark": -3.0, "smoke": -3.0, \
                           "flame": -4.0, "overgrown": -1.5, "tree": -1.0, "noise": -1.0, "sound": -1.0})
# Initilaize and load the SVM pickle
sgd = pickle.load(open('models/svm_model.pkl', 'rb'))


# Text analysis UI
user_input = st.text_area("Please describe the problem with the electric pole, and include any noticable visible or audible areas of concern.")

# Add an "Analyze" button for text sentiment analysis
if st.button('Submit'):
    # Perform sentiment analysis only if the button is pressed
    scores = sia.polarity_scores(user_input)
    topic = sgd.predict([user_input])
    sentiment = 'Neutral'
    if scores['compound'] > 0.05:
        sentiment = 'Positive'
    elif scores['compound'] < -0.05:
        sentiment = 'Negative'
    
    topic_mod = (topic[0]).replace("_", " ")
    # Display the sentiment result
    st.write(f"The overall predicted sentiment of this written report is: **{sentiment}** with a compound score of **{scores['compound']}**")
    st.write(f"The predicted topic of this written report is: **{topic_mod}**")
