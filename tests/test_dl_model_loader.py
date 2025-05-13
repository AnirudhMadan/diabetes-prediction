import unittest
from utils.dl_model_loader import load_dl_model_and_scaler
import os

class DLModelLoaderTestCase(unittest.TestCase):

    def test_model_loading(self):
        # Assuming model is saved as 'diabetes_model.h5'
        model_path = 'diabetes_model.h5'
        if os.path.exists(model_path):
            model, scaler = load_dl_model_and_scaler()
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            self.assertIn('predict', dir(model))
        else:
            self.skipTest(f"Model file {model_path} not found.")

if __name__ == "__main__":
    unittest.main()
