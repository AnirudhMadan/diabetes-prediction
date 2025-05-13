import unittest
from utils.qml import load_qml_model, predict_qml
import numpy as np

class QMLTestCase(unittest.TestCase):

    def test_qml_prediction(self):
        sample_input = np.array([5, 116, 74, 0, 0, 25.6, 0.201, 30]).reshape(1, -1)
        model = load_qml_model()
        result = predict_qml(model, sample_input)
        self.assertIn(result[0], [0, 1])  # Prediction should be either 0 or 1

if __name__ == "__main__":
    unittest.main()

