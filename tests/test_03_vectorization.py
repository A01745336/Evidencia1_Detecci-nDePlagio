import unittest
from plagiarism_detection import generate_vector_space_models

class TestVectorization(unittest.TestCase):
    def test_vector_output(self):
        texts = ["hello world", "hello"]
        vectors, feature_names = generate_vector_space_models(texts, texts)
        self.assertEqual(len(feature_names), 2)
        self.assertEqual(vectors.shape[0], 2)
        self.assertEqual(vectors.shape[1], 2)

    def test_empty_input(self):
        # Verifica que se maneje correctamente la entrada vacía
        with self.assertRaises(ValueError):
            generate_vector_space_models([], [])  # Entradas vacías



if __name__ == '__main__':
    unittest.main()
