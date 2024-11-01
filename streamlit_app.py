import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow.keras import backend as K
import os
from datetime import datetime



# Set the title of the web app
st.set_page_config(page_title="Flourish - Vegetation Analysis", layout="wide")

# Model paths
model_paths = {
    'U-Net': r"C:\Users\R\OneDrive\Desktop\Senior Project\unet_model(f).h5",
    'DeepLabV3+': r"C:\Users\R\OneDrive\Desktop\Senior Project\Fdeeplabv3+_model.keras",
    'PSPNet': r"C:\Users\R\OneDrive\Desktop\Senior Project\Fpspnet_model.keras",
}
# CSV file path for saving GVI results
csv_file_path = r"C:\Users\R\OneDrive\Desktop\Senior Project\GVI_results1.csv"
# Function to load selected model
def load_selected_model(selected_model):
    return load_model(
        model_paths[selected_model],
        custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou_metric': iou_metric},
        compile=False
    )

# Metrics for custom loss functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), tf.float32)
    y_pred_f = K.cast(K.flatten(y_pred), tf.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1]) 
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Image preprocessing and prediction
def preprocess_image(image):
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, (256, 256))
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function to generate green/black mask
def predict(image, model):
    processed_image = preprocess_image(tf.convert_to_tensor(np.array(image)))
    prediction = model.predict(processed_image)
    prediction = np.squeeze(prediction)  # Remove batch dimension

    # Create a color mask where:
    # 1 (greenery) -> [0, 255, 0] (green)
    # 0 (non-greenery) -> [0, 0, 0] (black)
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    color_mask[prediction > 0.5] = [0, 255, 0]  # Green for greenery
    color_mask[prediction <= 0.5] = [0, 0, 0]   # Black for non greenery

    return color_mask

# Streamlit page setup
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Main page 
def home_page():
    st.markdown("""<style>
        body {
            background-color: #E8F5E9;  /* Light Green Background */
            font-family: 'Roboto', sans-serif;
            color: #333333;
        }
        .header {
            text-align: center;
            padding: 20px;
            color: #2E7D32;  /* Darker Green for Header */
            font-size: 2.5em;
        }
        .sub-header {
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 40px;
            color: #388E3C;  /* Medium Green for Subheader */
        }
        .feature-card {
            background-color: #FFFFFF;  /* White Background for Cards */
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            color: #333333;
            height: 150px;
        }
        .feature-card:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);  /* Stronger Shadow on Hover */
        }
        .highlight {
            color: #FF5722;  /* Highlight Color */
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Flourish")
    st.markdown("<h2 class='sub-header'>Analyze Vegetation Levels with Flourish</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Flourish is a tool for evaluating vegetation levels, providing visual insights and data on greenery to aid environmental monitoring and landscape planning.</p>", unsafe_allow_html=True)

    # Create three equal columns for the cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class='feature-card'style='background-color: #E8F5E9;'>  <!-- Softer Green Background -->
                <h3 style='color: #2E7D32;'>Image Prediction🌿</h3>  <!-- Darker Green Text -->
                <p>Upload images and calculate the Green View Index (GVI) to assess vegetation.</p>
            </div>""", unsafe_allow_html=True)
        if st.button("Prediction", key="pred_button", use_container_width=True):
            st.session_state.page = "Prediction"

    with col2:
        st.markdown("""<div class='feature-card' style='background-color: #E8F5E9;'>  <!-- Softer Green Background -->
                <h3 style='color: #2E7D32;'>Data Visualization📊</h3>  <!-- Darker Green Text -->
                <p>View greenery levels on an interactive map and analyze patterns across locations.</p>
            </div>""", unsafe_allow_html=True)
        if st.button("Visualization", key="vis_button", use_container_width=True):
            st.session_state.page = "Visualization"

    with col3:
        st.markdown("""<div class='feature-card' style='background-color: #E8F5E9;'>  <!-- Softer Green Background -->
                <h3 style='color: #2E7D32;'>Add New Location📍</h3>  <!-- Darker Green Text -->
                <p>Input a new location's details, upload an image, and calculate the GVI.</p>
            </div>""", unsafe_allow_html=True)
        if st.button("Add Location", key="add_location_button", use_container_width=True):
            st.session_state.page = "Add Location"

def prediction_page():
    if st.button("< Back"):
        st.session_state.page = "Home"
        
    st.title("Image Prediction")
    st.write("Predict and Calculate the Green View Index (GVI) from Images")
    
    # Slider to select the number of images to upload
    num_images = st.slider("Select the number of images to upload:", min_value=1, max_value=10, value=4)

    st.markdown(f"<p class='highlight'>To calculate the Green View Index (GVI), please upload <strong>{num_images} images</strong> of the same location taken from different directions.</p>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model:", list(model_paths.keys()))
    model = load_selected_model(selected_model)

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) != num_images:
            st.error(f"Please upload exactly {num_images} images.")
        else:
            cols = st.columns(min(num_images, 4))  # Adjust the layout based on the number of images
            total_area = 0
            total_greenery_area = 0

            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                
                with cols[i % 4]:  
                    st.image(image.resize((150, 150)), caption="Original Image", use_column_width=True)  
                    prediction_image = Image.fromarray(predict(image, model))  # Use the color mask
                    st.image(prediction_image.resize((150, 150)), caption="Prediction Image", use_column_width=True, clamp=True)

                    total_image_area = prediction_image.size[0] * prediction_image.size[1]
                    greenery_area = np.sum(np.array(prediction_image)[:, :, 1] > 0)  # Count green pixels

                    total_area += total_image_area
                    total_greenery_area += greenery_area

            if total_area > 0:
                gvi = total_greenery_area / total_area
                gvi_percentage = gvi * 100  

                st.markdown(f"""
                <div style="padding: 10px; border: 2px solid green; background-color: #E8F5E9; border-radius: 10px; text-align: center;">
                    <h2 style="color: #388E3C;">Green View Index (GVI): {gvi_percentage:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("No data available to calculate GVI.")

def visualization_page():
    if st.button("< Back"):
        st.session_state.page = "Home"
        
    st.title("Data Visualization")
    file_path = r"C:\Users\R\OneDrive\Desktop\Senior Project\GVI_results1.csv"
    df = pd.read_csv(file_path)

    required_columns = ['Latitude', 'Longitude', 'greenry level']
    if all(col in df.columns for col in required_columns):
        st.write("Map showing Greenery Levels across locations:")

        color_map = {
            "Very Low Greenery": "#FF6347",
            "Low Greenery": "#FFA500",
            "Moderate Greenery": "#FFD700",
            "High Greenery": "#90EE90",
            "Very High Greenery": "#228B22"
        }
        
        df['Color'] = df['greenry level'].map(color_map)

        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="greenry level",
            color_discrete_map=color_map,
            hover_name="Date",
            mapbox_style="carto-positron",
            zoom=7,
            center={"lat": 21.059538, "lon": 41.251668},
            height=700
        )

        fig.update_layout(
            title="Greenery Levels Across Locations",
            title_x=0.4,
            title_font_size=24,
            margin={"r":0,"t":40,"l":0,"b":0},
            paper_bgcolor='#E8F5E9',
            plot_bgcolor='#121212',
            font=dict(color="#FFFFFF")
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("The columns are not in the CSV file.")

def add_location_page():
    if st.button("< Back"):
        st.session_state.page = "Home"
    
    st.title("Add New Location")
    st.write("Input location details and upload 4 images in same location taken from different directions to add a new location.")
    
    # Location input form
    with st.form("Add Location Form"):
        latitude = st.number_input("Latitude", format="%.6f")
        longitude = st.number_input("Longitude", format="%.6f")
        province = st.text_input("Province")
        
        # File uploader for exactly 4 images
        uploaded_images = st.file_uploader("Choose 4 images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        selected_model = st.selectbox("Select Model:", list(model_paths.keys()))
        submitted = st.form_submit_button("Upload and Predict GVI")
        
        # Check if the form is submitted and the required conditions are met
        if submitted and uploaded_images and province:
            if len(uploaded_images) != 4:
                st.error("Please upload exactly 4 images.")
                return
            
            model = load_selected_model(selected_model)
            total_image_area = 0
            total_greenery_area = 0
            
            # Create directory for saving images if it doesn't exist
            save_directory = r"C:\Users\R\OneDrive\Desktop\Senior Project\add_new_location"
            os.makedirs(save_directory, exist_ok=True)
            
            # Get the current date in the required format
            current_date = datetime.now().strftime('%Y-%m')
            
            # Process each uploaded image
            for i, uploaded_image in enumerate(uploaded_images):
                image = Image.open(uploaded_image)
                
                # Format the filename with the specified format
                image_filename = os.path.join(
                    save_directory,
                    f"image_{str(latitude).replace('.', '-')}_{str(longitude).replace('.', '-')}_{i}_{current_date}.jpg"
                )
                
                # Save the uploaded image to the specified directory
                image.save(image_filename)
                
                # Predict for each image
                prediction_image = predict(image, model)
                image_area = prediction_image.shape[0] * prediction_image.shape[1]
                greenery_area = np.sum(prediction_image[:, :, 1] > 0)
                
                total_image_area += image_area
                total_greenery_area += greenery_area
            
            # Calculate GVI and GVI ratio
            gvi = total_greenery_area / total_image_area
            gvi_ratio = (total_greenery_area / total_image_area) * 100
            format_gvi_ratio = f"{gvi_ratio:.2f}%"  # Format GVI Ratio to two decimal places with %

            # Determine greenery level
            if gvi_ratio < 5:
                greenery_level = "Very Low"
            elif gvi_ratio < 15:
                greenery_level = "Low"
            elif gvi_ratio < 25:
                greenery_level = "Normal"
            elif gvi_ratio < 35:
                greenery_level = "High"
            else:
                greenery_level = "Very High"            
            # Display GVI result
            st.markdown(f"""
            <div style="padding: 10px; border: 2px solid green; background-color: #E8F5E9; border-radius: 10px; text-align: center;">
                <h2 style="color: #388E3C;">Green View Index (GVI): {format_gvi_ratio}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Append data to CSV with additional fields
            new_data = pd.DataFrame({
                "Latitude": [latitude],
                "Longitude": [longitude],
                "Date": [current_date],  # Date format changed to YYYY-MM
                "Total_Greenery_Area": [total_greenery_area],
                "Total_Area": [total_image_area],
                "GVI": [gvi],
                "GVI Ratio": [format_gvi_ratio],
                "greenry level": [greenery_level],
                "province": [province]
            })
            new_data.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
            st.success("New location added successfully.")

            

# Define the Power BI URL
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiODU4YjI0NDYtMjQ3Ni00M2FkLTlkYTItNjMzZjA5MWNjZDJiIiwidCI6ImI0NTNkOTFiLTZhYzEtNGI2MS1iOGI4LTVlNjVlNDIyMjMzZiIsImMiOjl9"

# Styled Markdown button that opens the Power BI report directly
st.markdown(f'''
    <a href="{power_bi_url}" target="_blank">
        <button style="padding:10px 20px; font-size:16px; color:white; background-color:#4CAF50; border:none; border-radius:5px; cursor:pointer;">
            Go to Visualization
        </button>
    </a>
''', unsafe_allow_html=True)

# Main application 
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Prediction":
    prediction_page()
elif st.session_state.page == "Visualization":
    visualization_page()
elif st.session_state.page == "Add Location":
    add_location_page()
