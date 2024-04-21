import unittest
from plagiarism_detection import evaluate_performance, generate_report
from sklearn.metrics import roc_curve, auc

class TestEvaluation(unittest.TestCase):
    def test_performance_metrics(self):
        similarities = [0.5, 0.2, 0.9]
        ground_truth = [1, 0, 1]
        results = evaluate_performance(similarities, 0.3, ground_truth)
        self.assertEqual(results['TP'], 2)
        self.assertEqual(results['TN'], 1)
        self.assertEqual(results['FP'], 0)
        self.assertEqual(results['FN'], 0)

    def test_auc_calculation(self):
        # AUC should be correctly calculated
        similarities = [0, 0.5, 1]
        ground_truth = [0, 1, 1]
        fpr, tpr, thresholds = roc_curve(ground_truth, similarities)
        auc_value = auc(fpr, tpr)
        self.assertAlmostEqual(auc_value, 0.75)

    def test_mismatched_lengths(self):
        # Verifica que se maneje correctamente cuando las longitudes no coinciden
        similarities = [0.5]  # Solo un elemento
        ground_truth = { 'doc1.txt': True, 'doc2.txt': False }  # Dos elementos
        with self.assertRaises(ValueError):
            evaluate_performance(similarities, 0.5, ground_truth)

if __name__ == '__main__':
    unittest.main()
