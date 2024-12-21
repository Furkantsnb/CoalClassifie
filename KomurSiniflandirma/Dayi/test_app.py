import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app import load_model, process_image, predict_image


@patch("streamlit.file_uploader")
def test_load_model(mock_file_uploader):

    mock_file_uploader.return_value = "dummy_file_path"
    

    valid_model_path = "efficent_net_efficent224.h5"
    

    loaded_model = load_model(valid_model_path)
    

    assert loaded_model is not None

def test_process_image():

    with patch("tensorflow.keras.preprocessing.image.load_img") as mock_load_img, \
         patch("tensorflow.keras.preprocessing.image.img_to_array") as mock_img_to_array:

        mock_load_img.return_value = MagicMock()
        mock_img_to_array.return_value = MagicMock()
        
        processed_image = process_image("dummy_image_path")
        
        assert processed_image is not None

def test_predict_image():

    model_mock = MagicMock()
    img_mock = MagicMock()

    model_mock.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])

    predicted_class, prediction = predict_image(model_mock, img_mock)
    
    assert predicted_class == 3
    
    assert prediction is not None
