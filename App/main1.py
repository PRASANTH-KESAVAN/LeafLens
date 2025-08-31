import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time




# Apply custom background color















# Load the plant disease details CSV
disease_details_df = pd.read_csv('plant_disease_details.csv')

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element










# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])



# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "image.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. Train (70,295 images)
                2. Test (33 images)
                3. Validation (17,572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    
    # Predict button
    if st.button("Predict"):
        # st.snow()
        with st.spinner('Processing your request...'):
           time.sleep(5)
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        disease_name = class_name[result_index]
        # Custom-styled success message
        st.markdown(
            f"""
            <div style="
                background-color: #d4edda; 
                color: #000000; 
                border-left: 5px solid #28a745; 
                padding: 10px; 
                border-radius: 5px; 
                font-size: 24px; 
                font-weight: bold;">
                Model is predicting it's a {disease_name}
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Get disease details from CSV based on predicted disease
        disease_details = disease_details_df[disease_details_df['Plant Disease'] == disease_name].iloc[0]

        # Display disease details
        st.subheader(f"Details for {disease_name}:")
        st.markdown("To know more visit 'Indian council of Agricultural Resource'")
        st.markdown(
    '<a href="https://www.icar.org.in/" target="_blank" style="text-decoration: none;">'
    '<button style="background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">'
    'Visit ICAR'
    '</button>'
    '</a>',
    unsafe_allow_html=True
)


        st.write(f"**Recommended Fertilizers:** {disease_details['Recommended Fertilizer']}")
        st.markdown(
    '<a href="https://www.fertilizer.org/about-fertilizers/what-are-fertilizers/" target="_blank" style="text-decoration: none;">'
    '<button style="background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">'
    'Visit IFA'
    '</button>'
    '</a>',
    unsafe_allow_html=True
)


        st.write(f"**Precautions:** {disease_details['Precautions']}")
        st.write(f"**Causes:** {disease_details['Causes']}")
        st.write(f"**Best Climate:** {disease_details['Best Climates']}")
        st.write(f"**Worst Climate:** {disease_details['Worst Climates']}")
        st.write(f"**Best Period (Months):** {disease_details['Best Period']}")
        st.write(f"**Worst Period (Months):** {disease_details['Worst Period']}")


        





