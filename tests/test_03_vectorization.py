import unittest
from plagiarism_detection import generate_vector_space_models

class TestVectorization(unittest.TestCase):
    def test_vector_output(self):
        texts = ["hello world", "hello"]
        original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(texts, texts)
        # Ahora esperamos 3 características: 'hello', 'world', y 'hello world'
        self.assertEqual(len(feature_names), 3)
        self.assertEqual(original_vectors.shape[0], 2)
        self.assertEqual(suspicious_vectors.shape[0], 2)
        self.assertEqual(original_vectors.shape[1], 3)
        self.assertEqual(suspicious_vectors.shape[1], 3)

    def test_empty_input(self):
        # Verifica que se maneje correctamente la entrada vacía
        with self.assertRaises(ValueError):
            generate_vector_space_models([], [])  # Entradas vacías



if __name__ == '__main__':
    unittest.main()
