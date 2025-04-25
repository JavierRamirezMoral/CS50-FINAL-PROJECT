import pytest
import os
import numpy as np
from project import (
    load_and_preprocess_image,
    classify_image,
    display_predictions,
)

# Test images paths - UPDATE THESE TO MATCH YOUR ACTUAL PATHS
TEST_IMAGES = {
    "cat": r"C:\Users\....CS50P_Final_Project\assets\animals\cat.jpg",
    "dog": r"C:\Users\....CS50P_Final_Project\assets\animals\dog.jpg",
    "clownfish": r"C:\Users\....CS50P_Final_Project\assets\animals\clownfish.jpg",
    "pizza": r"C:\Users\....CS50P_Final_Project\assets\food\pizza.jpg",
    "apple": r"C:\Users\....CS50P_Final_Project\assets\food\apple.jpg",
    "icecream": r"C:\Users\....CS50P_Final_Project\assets\food\icecream.jpg",
    "car": r"C:\Users\....CS50P_Final_Project\assets\objects\car.jpg",
    "cup": r"C:\Users\....CS50P_Final_Project\assets\objects\cup.jpg",
    "pc": r"C:\Users\....CS50P_Final_Project\assets\objects\pc.jpg"
}


# Setup and teardown for CSV tests
@pytest.fixture
def cleanup_csv():
    yield
    # Clean up after CSV tests
    if os.path.exists("test_results.csv"):
        os.remove("test_results.csv")

def test_load_and_preprocess_image():
    """Test that image loading and preprocessing works correctly"""
    img_path = TEST_IMAGES["cat"]
    img_array = load_and_preprocess_image(img_path)
    
    assert img_array is not None
    assert isinstance(img_array, np.ndarray)
    assert img_array.shape == (1, 224, 224, 3)  # Batch of 1, 224x224, 3 channels
    
    # Test with non-existent image - modified to expect None return
    result = load_and_preprocess_image("non_existent.jpg")
    assert result is None

def test_classify_image():
    """Test that image classification returns expected results"""
    img_path = TEST_IMAGES["dog"]
    img_array = load_and_preprocess_image(img_path)
    predictions = classify_image(img_array)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 3  # Top 3 predictions
    for pred in predictions:
        assert len(pred) == 3  # (id, label, probability)
        assert isinstance(pred[0], str)  # ID
        assert isinstance(pred[1], str)  # Label
        # Changed to check for numpy float type instead of Python float
        assert isinstance(pred[2], (float, np.float32))  # Probability
        assert 0 <= pred[2] <= 1  # Probability between 0 and 1

def test_display_predictions():
    """Test the display formatting of predictions"""
    mock_predictions = [
        ("n02123159", "tiger_cat", 0.8765),
        ("n02123045", "tabby", 0.1234),
        ("n02124075", "egyptian_cat", 0.0001)
    ]
    result = display_predictions(mock_predictions)
    
    assert isinstance(result, str)
    lines = result.split("\n")
    assert len(lines) == 3
    assert "Tiger Cat (87.65%)" in lines[0]
    assert "Tabby (12.34%)" in lines[1]
    assert "Egyptian Cat (0.01%)" in lines[2]
