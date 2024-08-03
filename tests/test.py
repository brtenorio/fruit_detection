import pytest 
from fruit_detection.evaluate_model import evaluate_model
from fruit_detection.data_generator import test_generator

def test_model():
    eval = evaluate_model(test_generator)
    print('Test Accuracy: ', eval[1])
    try: 
        assert eval[1] > 0.75
    except:
        raise Exception("test failed. Try rebuilding the model.")

