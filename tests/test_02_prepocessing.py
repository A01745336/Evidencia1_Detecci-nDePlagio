import unittest
from plagiarism_detection import preprocess

class TestPreprocessing(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(preprocess("Test"), "test")

    def test_remove_punctuation(self):
        self.assertEqual(preprocess("Hello, world!"), "hello world")

    def test_stemming(self):
        self.assertIn("run", preprocess("running"))

    def test_non_string_input(self):
        # Verifica que se maneje correctamente una entrada que no es un string
        with self.assertRaises(AttributeError):
            preprocess(123)  # Pasando un número en lugar de un string

if __name__ == '__main__':
    unittest.main()
