import unittest
from plagiarism_detection import read_files_in_directory

class TestFileReading(unittest.TestCase):
    def test_empty_directory(self):
        filenames, contents = read_files_in_directory('./empty')
        self.assertEqual(len(filenames), 0)
        self.assertEqual(len(contents), 0)

    def test_nonexistent_directory(self):
        with self.assertRaises(Exception):
            read_files_in_directory('path/to/nonexistent/directory')

    def test_reading_contents(self):
        filenames, contents = read_files_in_directory('./TextosConPlagio')
        self.assertGreater(len(filenames), 0)
        self.assertEqual(len(filenames), len(contents))

if __name__ == '__main__':
    unittest.main()
