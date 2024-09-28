import unittest
from src.dataReuploading import SingleQNN
from src.Sampler import Sampler
class SingleQNNTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_set, cls.train_label = Sampler().circle(n_points=100)
        cls.test_set, cls.test_label = Sampler().circle(n_points=20)

        cls.qnn = SingleQNN(num_layers=7)
    def test_train(self):
        df = self.qnn.train(self.train_set, self.train_label, self.test_set, self.test_label)
        print(df)

if __name__ == '__main__':
    unittest.main()