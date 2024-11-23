import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img
from components import Models
import io
import time
import tempfile


def main():
    model=Models()
    # Set page configuration
    st.set_page_config(
        page_title="dog breed classifier",
        page_icon="📷",
        layout="wide"
    )
    
    # Add title and description
    st.title("📷 dog breed classifier")
    st.markdown("""
    Upload an image and convert it to text.
    Supported formats: PNG, JPG, JPEG
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Convert the uploaded file to bytes
            bytes_data = uploaded_file.getvalue()
            
            # Create a BytesIO object
            image_bytes = io.BytesIO(bytes_data)
            
            # Open the image using PIL
            try:
                image = Image.open(image_bytes)
                st.image(image, caption='DOG IMAGE', use_container_width=True)
                
                # Add process button
                if st.button("Classify", type="primary"):
                    # Show progress
                    with st.spinner('Processing image...'):
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format="JPEG")
                        img_byte_arr.seek(0)
                        # img_byte_arr = io.BytesIO()
                        # image.save(img_byte_arr, format="JPEG")
                        # img_byte_arr.seek(0)

                        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
                            tmp_file.write(img_byte_arr.read())
                            tmp_file.flush()
                            # Load the image using `load_img`
                            img_g = load_img(tmp_file.name,target_size = model.img_size)


                        # Pass BytesIO object to `load_img`
                        bread,accuracy=model.predict(img_g)
                        # Simulate processing time
                        progress_bar = st.progress(0)
                        for i in range(100):
                            
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                            
                        # Here you would normally integrate with an OCR service
                        # For demo purposes, we'll just show a placeholder text
                        with col2:
                            st.subheader("OUTPUT")
                            st.text_area(
                                "Detected bread of the dog:",
                                value="Bread: "+bread,
                                      
                                height=100
                            )
                            
                            # Add download button for the extracted text
                            
                            # Add confidence score (simulated)
                            st.metric(
                                label="Confidence Score",
                                value=f"{accuracy}%",
                            )
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Add information about supported formats and limitations
   
if __name__ == "__main__":
    main()