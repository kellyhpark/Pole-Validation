import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from streamlit_image_select import image_select

# Streamlit UI
st.title('CERA: ML Model Demo')

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
    
# Image selection
img = image_select(
    label="Select an example image",
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
    st.write("**An electric pole has been identified with 58% confidence**")
elif img == "data/validation/IMG_6864.jpg":
    st.image("models/detr_results/valid012.png")
    st.write("**An electric pole has been identified with 73% confidence**")
elif img == "data/validation/IMG_6875.jpg":
    st.image("models/detr_results/valid0122.png")
    st.write("**An electric pole has been identified with 68% confidence**")
else:
    st.image("models/detr_results/valid0123.png")
    st.write("**An electric pole has been identified with 97% confidence**")


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
    
    # Display the sentiment result
    st.write(f"Sentiment: **{sentiment, scores['compound']}**")
    st.write(f"The topic is **{topic[0]}**")
