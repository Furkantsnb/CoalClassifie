import pytest
import tensorflow as tf
from main import dataset_loading, build_efficentB0

@pytest.fixture
def sample_data():
    train_dataset, validation_dataset, test_dataset = dataset_loading()
    return train_dataset, validation_dataset, test_dataset

def test_dataset_loading(sample_data):
    train_dataset, validation_dataset, test_dataset = sample_data
    assert isinstance(train_dataset, tf.data.Dataset)
    assert isinstance(validation_dataset, tf.data.Dataset)
    assert isinstance(test_dataset, tf.data.Dataset)

def test_build_efficentB0():
    model = build_efficentB0(num_classes=4)
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0

def test_model_training(sample_data):
    train_dataset, validation_dataset, _ = sample_data
    model = build_efficentB0(num_classes=4)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=1)
    assert isinstance(history, tf.keras.callbacks.History)
    assert 'accuracy' in history.history
    assert 'val_accuracy' in history.history

def test_model_evaluation(sample_data):
    _, _, test_dataset = sample_data
    model = build_efficentB0(num_classes=4)
    model.evaluate(test_dataset)
