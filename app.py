import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('model.h5')

# Function to preprocess image
def preprocess_image(image):
    img_array = np.array(image)
    rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array

# Function to make prediction
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

# Main page
def show_main_page():
    st.title("Alzheimer's Disease Detection")
    st.markdown("Please upload an image.")
    uploaded_file = st.file_uploader(label="", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predicted_idx = predict(image)
        class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
        predicted_label = class_labels[predicted_idx]
        st.markdown(f"<p class='prediction' style='color:red; text-align:center; font-weight: bold; font-size:24px; background-color: #f0f6fc; padding: 20px; border-radius: 10px;'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)
    st.markdown("""
        <div>
            <p></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Button to go back to landing page
    if st.button(" &#9664; Back to Landing Page" ):
        st.session_state.page = 'landing'
    st.markdown("---")

# Landing page
def show_landing_page():
    st.title("Alzheimer's Disease Detection")
    st.markdown("""
        <div style='background-color: #f0f6fc; padding: 20px; border-radius: 10px;'>
            <h2>About Alzheimer's Disease</h2>
            <p>Alzheimer's disease is a progressive neurological disorder characterized by the gradual degeneration of brain cells, leading to a decline in cognitive function and memory. This condition, accounting for a significant portion of dementia cases, manifests through various symptoms, including memory loss, confusion, and difficulties with problem-solving and language. As the disease progresses, individuals may experience changes in behavior and personality, along with challenges in completing daily tasks. Despite ongoing research efforts, the exact cause of Alzheimer's remains elusive, with factors such as genetics, lifestyle, and environmental influences playing a role. Currently, there is no cure for Alzheimer's disease, but treatments aim to manage symptoms and improve quality of life for affected individuals and their caregivers. Enhancing awareness, supporting research initiatives, and providing comprehensive care are essential components in addressing the impact of Alzheimer's disease on individuals and communities.</p>           
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: #f0f6fc; padding: 20px; border-radius: 10px;'>
            <h2>Proposed Solution: CNN</h2>
            <p>I proposed a solution using Convolutional Neural Networks (CNNs) for Alzheimer's disease detection 
            based on brain ultrasound images. CNNs have shown promising results in image recognition tasks and 
            can effectively learn features from medical images.</p>
            
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #f0f6fc; padding: 20px; border-radius: 10px;'>        
            <h3>Performance Details</h3>
            <ul>
                <li>Test Loss: 0.035</li>
                <li>Test Accuracy: 99.07%</li>
                <li>Test AUC: 99.97%</li>
                <li>Test Precision: 99.07%</li>
                <li>Test Recall: 99.07%</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div>
            <p></p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div>
            <p></p>
        </div>
    """, unsafe_allow_html=True)
    # Positioning the "Try the App" button on the right side
    col1, col2 = st.columns([1, 2])
    with col2:
        if st.button("Try the App >>"):
            st.session_state.page = 'main'
    st.markdown("---")

# Contact us section
def show_contact_section():
    st.title("Developer Contact:")
    st.markdown("""
        <div style='background-color: #f0f6fc; padding: 20px; border-radius: 10px;'>
            <h3>For any queries or feedback, please contact developer:</h3>
            <p>Name: P Priya</p>
            <p>Email: priya010322@gmail.com</p>
            <p>Phone: 9025078585</p>
        </div>
    """, unsafe_allow_html=True)

# Render the app
if __name__ == '__main__':
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    
    if st.session_state.page == 'main':
        show_main_page()
    else:
        show_landing_page()
    
    show_contact_section()
    st.markdown('</div>', unsafe_allow_html=True)
