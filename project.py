# Standard library imports
import os
import csv
from datetime import datetime
import webbrowser

# Third-party library imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import gradio as gr

# ======================
# CONFIGURATION SECTION
# ======================

# Configure environment to suppress TensorFlow logs and warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 3=ERROR only, 2=INFO, 1=WARNING, 0=DEBUG
tf.get_logger().setLevel('ERROR')           # Sets TensorFlow logger to only show errors

# ======================
# MODEL INITIALIZATION
# ======================

# Load MobileNetV2 model with ImageNet weights (only once for efficiency)
MODEL = MobileNetV2(weights="imagenet")

# ======================
# CORE FUNCTIONS
# ======================

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads and preprocesses an image for the model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array ready for model prediction or None if error occurs
        
    Technical Details:
        - Uses Keras' image.load_img() to load image in 224x224 resolution (MobileNetV2 input size)
        - Converts image to numpy array and adds batch dimension
        - Applies MobileNetV2-specific preprocessing (normalization)
    """
    try:
        # Load image with target size for MobileNetV2
        img = image.load_img(image_path, target_size=(224, 224))
        # Convert image to numpy array
        img_array = image.img_to_array(img)
        # Add batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # Apply model-specific preprocessing
        return preprocess_input(img_array)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def classify_image(img_array: np.ndarray) -> list:
    """
    Classifies an image using the preloaded MobileNetV2 model.
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        List of top 3 predictions with (class_id, class_name, probability)
        
    Technical Details:
        - Uses MobileNetV2's predict() method
        - decode_predictions() converts numerical outputs to human-readable classes
        - Returns top 3 predictions by probability
    """
    predictions = MODEL.predict(img_array, verbose=0)
    return decode_predictions(predictions, top=3)[0]


def display_predictions(predictions: list) -> str:
    """
    Formats prediction results for display.
    
    Args:
        predictions: List of predictions from classify_image()
        
    Returns:
        Formatted string with ranking, class names, and probabilities
        
    Technical Details:
        - Processes each prediction tuple (id, label, probability)
        - Replaces underscores with spaces and capitalizes words
        - Formats probabilities as percentages with 2 decimal places
    """
    results = []
    for i, (_, label, prob) in enumerate(predictions):
        results.append(f"{i+1}. {label.replace('_', ' ').title()} ({prob*100:.2f}%)")
    return "\n".join(results)

# ======================
# UTILITY FUNCTIONS
# ======================

def save_to_csv(image_path: str, predictions: list, filename: str = "results.csv") -> None:
    """
    Saves classification results to a CSV file in Excel-friendly format.
    
    Args:
        image_path: Original path to the image file
        predictions: List of predictions from classify_image()
        filename: Output CSV filename
        
    Technical Details:
        - Uses utf-8-sig encoding for Excel compatibility
        - CSV writer configured with minimal quoting
        - Only writes header if file doesn't exist or is empty
        - Extracts basename from path for cleaner display
    """
    # Prepare data
    file_exists = os.path.isfile(filename)
    short_name = os.path.basename(image_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format predictions
    formatted_preds = [
        f"{label.replace('_', ' ').title()} ({prob*100:.2f}%)"
        for _, label, prob in predictions
    ]
    
    # Write to CSV with Excel-optimized settings
    with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        # Write header if needed
        if not file_exists or os.stat(filename).st_size == 0:
            writer.writerow([
                'Date', 
                'Image Name', 
                'Top Prediction', 
                'Second Prediction', 
                'Third Prediction'
            ])
        
        # Write data row
        writer.writerow([timestamp, short_name, *formatted_preds])

# ======================
# GUI INTERFACE
# ======================

def gradio_interface():
    """
    Creates an enhanced GUI interface using Gradio.
    
    Technical Details:
        - Uses Gradio Blocks for flexible layout
        - Custom CSS for professional styling
        - Two-column layout (input/output)
        - Example images for quick testing
        - Automatic browser opening
        - Emoji-enhanced results display
    """
    # Custom CSS styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #4a6fa5;
    }
    .description {
        text-align: center;
        margin-bottom: 20px;
    }
    .results-box {
        padding: 15px;
        border-radius: 10px;
        background: #f5f7fa;
        margin-top: 20px;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        # Title and description
        gr.Markdown("""
        <div class="title">
            <h1>🖼️ AI Image Classifier</h1>
        </div>
        <div class="description">
            <p>Upload an image to classify using the pre-trained MobileNetV2 model</p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                # Image input component
                image_input = gr.Image(type="filepath", label="Image to Classify", 
                                     elem_id="upload-box")
                
                # Submit button
                submit_btn = gr.Button("Classify Image", variant="primary")
                
                # Example images
                gr.Examples(
                    examples=[
                        os.path.join("assets", "animals", "cat.jpg"),
                        os.path.join("assets", "food", "pizza.jpg"),
                        os.path.join("assets", "objects", "pc.jpg")
                    ],
                    inputs=image_input,
                    label="Examples - Click any image to test"
                )

            with gr.Column():
                # Uploaded image preview
                image_output = gr.Image(label="Uploaded Image", interactive=False)
                
                # Results in styled container
                with gr.Column(elem_classes="results-box"):
                    gr.Markdown("### 🔍 Classification Results")
                    results_output = gr.Textbox(label="Predictions", interactive=False)
                
                # CSV history button
                csv_btn = gr.Button("View Full History", variant="secondary")
                csv_btn.click(
                    fn=lambda: webbrowser.open(os.path.abspath("results.csv")),
                    inputs=None,
                    outputs=None
                )

        def predict(image_path: str) -> tuple:
            """
            Processing function for image classification.
            
            Args:
                image_path: Path to the uploaded image
                
            Returns:
                tuple: (image_path, formatted_results)
            """
            if image_path is None:
                return None, "Please upload an image"
            
            img_array = load_and_preprocess_image(image_path)
            predictions = classify_image(img_array)
            save_to_csv(image_path, predictions)
            
            # Format results with emojis
            formatted_results = []
            emojis = ["🥇", "🥈", "🥉"]
            for i, (_, label, prob) in enumerate(predictions):
                formatted_results.append(
                    f"{emojis[i]} {label.replace('_', ' ').title()}: {prob*100:.2f}%"
                )
            
            return image_path, "\n".join(formatted_results)

        # Connect components
        submit_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[image_output, results_output]
        )

        # Open browser automatically
        webbrowser.open("http://127.0.0.1:7860")

    return demo
# ======================
# BATCH PROCESSING FUNCTION
# ======================

def batch_process(image_paths: list) -> dict:
    """
    Processes multiple images in batch mode and returns classification results.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        Dictionary with:
        - keys: image paths
        - values: formatted prediction strings
        
    Technical Details:
        - Processes each image sequentially
        - Uses MobileNetV2 for classification
        - Skips unprocessable images
        - Returns results in memory without file I/O
    """
    results = {}
    
    for img_path in image_paths:
        try:
            # Load and preprocess image
            img_array = load_and_preprocess_image(img_path)
            if img_array is None:
                continue
                
            # Classify image and store results
            predictions = classify_image(img_array)
            results[img_path] = display_predictions(predictions)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return results

# ======================
# MAIN APPLICATION
# ======================

def main():
    """
    Main entry point for the application.
    
    Technical Details:
        - Clears console before starting
        - Provides CLI and GUI modes
        - Handles user input validation
        - Manages the application flow
    """
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🖼️ AI Image Classifier")
    mode = input("CLI or GUI mode? (cli/gui): ").strip().lower()

    if mode == "cli":
        # Command Line Interface mode
        image_path = input("Image path (e.g. 'assets/dog.jpg'): ")
        img_array = load_and_preprocess_image(image_path)
        if img_array is not None:
            predictions = classify_image(img_array)
            print("\n🔍 Results:")
            print(display_predictions(predictions))
            save_to_csv(image_path, predictions)
    elif mode == "gui":
        # Graphical User Interface mode
        gradio_interface().launch(
            share=False,      # Don't create public link
            show_error=True, # Show errors in interface
            quiet=True       # Suppress Gradio logs
        )
    else:
        print("❌ Invalid option.")

if __name__ == "__main__":
    # Entry point when script is run directly
    main()