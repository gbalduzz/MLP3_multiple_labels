#from .. import modules
import modules.kernels as krn
import unittest

class kernel_test(unittest.TestCase):
    def min_hist_test(self):
        a = [1,3,6]
        b = [3,3,1]
        expected = 1+3+1
        self.assertEqual(expected, krn.min_isto(a,b))
    def min_hist2(self):
        a = [[1, 3, 6], [1, 3, 6]]
        b = [[3, 3, 1], [3, 3, 1]]
        res = krn.min_isto(a,b)
        self.assertEqual([5,5], res)

if __name__ == '__main__':
    unittest.main()
