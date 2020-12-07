import unittest
from ActiveAnnotation import ActiveAnnotation
import random

class TestStringMethods(unittest.TestCase):
    
    def setUp(self):
        self.data = [str(i) for i in range(0,100)]
        self.an = ActiveAnnotation(self.data, lambda x: x)

    def test_length_(self):
        self.an._negative_sample(n=10)
        self.assertEqual(len(self.an.XtrainNeg), 10, "Length incorrect")
    
    def test_heristic_labelling(self):
        

if __name__ == '__main__':
    unittest.main()