import unittest
import numpy as np
try:
    from readData import readData
except ImportError:
    from pyees.readData import readData

class test(unittest.TestCase):

    def testReadFileTypes(self):
        # xlsx file
        dat = readData('testData/data1.xlsx', 'A-B')
        np.testing.assert_array_equal(dat.s1.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.s1.a.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.s1.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat.s1.b.uncert, [0, 0, 0, 0, 0])

        # xls file
        dat = readData('testData/data1.xls', 'A-B')
        np.testing.assert_array_equal(dat.s1.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.s1.a.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.s1.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat.s1.b.uncert, [0, 0, 0, 0, 0])

        # csv file
        with self.assertRaises(Exception) as context:
            dat = readData('testData/data1.csv', 'A-B')
        self.assertTrue("The file extension is not supported. The supported extension are ['.xls', '.xlsx']" in str(context.exception))

    def testReadUncertanty(self):
        dat = readData('testData/data3.xlsx', 'A-B', 'C-D')
        np.testing.assert_array_equal(dat.s1.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.s1.a.unit), 'm')
        np.testing.assert_array_equal(dat.s1.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat.s1.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat.s1.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])

        dat = readData('testData/data4.xlsx', 'A-B', 'C-D')
        np.testing.assert_array_equal(dat.s1.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.s1.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_almost_equal([elem[6] for elem in dat.s1.a.covariance[dat.s1.b]][0], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        np.testing.assert_array_equal(dat.s1.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat.s1.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elem[6] for elem in dat.s1.b.covariance[dat.s1.a]][0], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)

        with self.assertRaises(Exception) as context:
            dat6 = readData('testData/data6.xlsx', 'A-B', 'C-D')
        self.assertTrue("The covariances has to be symmetric" in str(context.exception))

    def testAppend(self):
        dat1 = readData('testData/data1.xlsx', 'A-B')
        dat2 = readData('testData/data2.xlsx', 'A-B')
        dat1.s1.append(dat2.s1)
        np.testing.assert_array_equal(dat1.s1.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat1.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat1.s1.a.uncert, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat1.s1.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat1.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat1.s1.b.uncert, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        dat4 = readData('testData/data4.xlsx', 'A-B')
        dat5 = readData('testData/data5.xlsx', 'A-B')
        with self.assertRaises(Exception) as context:
            dat4.s1.append(dat5.s1)
        self.assertTrue("You can only append sheets with the excact same measurements. The names did not match" in str(context.exception))

        dat3 = readData('testData/data3.xlsx', 'A-B')
        dat4 = readData('testData/data4.xlsx', 'A-B')
        with self.assertRaises(Exception) as context:
            dat3.s1.append(dat4.s1)
        self.assertTrue("You can not set an element of [1, 2, 3, 4, 5] [m] with [1, 2, 3, 4, 5] [L/min] as they do not have the same unit" in str(context.exception))

        dat2 = readData('testData/data2.xlsx', 'A-B', 'C-D')
        dat4 = readData('testData/data4.xlsx', 'A-B', 'C-D')
        dat2.s1.append(dat4.s1)
        np.testing.assert_array_equal(dat2.s1.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat2.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat2.s1.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat2.s1.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat2.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat2.s1.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elem[6] for elem in dat2.s1.a.covariance[dat2.s1.b]][0], np.array([0, 0, 0, 0, 0, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 1000)
        np.testing.assert_almost_equal([elem[6] for elem in dat2.s1.b.covariance[dat2.s1.a]][0], np.array([0, 0, 0, 0, 0, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 1000)

        dat2 = readData('testData/data2.xlsx', 'A-B', 'C-D')
        dat4 = readData('testData/data4.xlsx', 'A-B', 'C-D')
        dat4.s1.append(dat2.s1)
        np.testing.assert_array_equal(dat4.s1.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat4.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat4.s1.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat4.s1.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat4.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat4.s1.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elem[6] for elem in dat4.s1.a.covariance[dat4.s1.b]][0], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0, 0, 0, 0, 0]) / 1000 / 60 / 1000)
        np.testing.assert_almost_equal([elem[6] for elem in dat4.s1.b.covariance[dat4.s1.a]][0], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0, 0, 0, 0, 0]) / 1000 / 60 / 1000)

        dat4_1 = readData('testData/data4.xlsx', 'A-B', 'C-D')
        dat4_2 = readData('testData/data4.xlsx', 'A-B', 'C-D')
        dat4_1.s1.append(dat4_2.s1)
        np.testing.assert_array_equal(dat4_1.s1.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat4_1.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat4_1.s1.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat4_1.s1.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat4_1.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat4_1.s1.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elem[6] for elem in dat4_1.s1.a.covariance[dat4_1.s1.b]][0], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 6000)
        np.testing.assert_almost_equal([elem[6] for elem in dat4_1.s1.b.covariance[dat4_1.s1.a]][0], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 6000)

    def testIndex(self):
        dat = readData('testData/data1.xlsx', 'A-B')
        np.testing.assert_array_equal(dat.s1.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.s1.a.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.s1.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat.s1.b.uncert, [0, 0, 0, 0, 0])

        dat.s1 = dat.s1[0:3]
        np.testing.assert_array_equal(dat.s1.a.value, [1, 2, 3])
        self.assertEqual(str(dat.s1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.s1.a.uncert, [0, 0, 0])
        np.testing.assert_array_equal(dat.s1.b.value, [5, 6, 7])
        self.assertEqual(str(dat.s1.b.unit), 'mA')
        np.testing.assert_array_equal(dat.s1.b.uncert, [0, 0, 0])

    def testIterable(self):
        # xlsx file
        dat = readData('testData/data1.xlsx', 'A-B')
        for sheet in dat:
            for i, meas in enumerate(sheet):
                if i == 0:
                    np.testing.assert_array_equal(meas.value, [1, 2, 3, 4, 5])
                    self.assertEqual(str(meas.unit), 'L/min')
                    np.testing.assert_array_equal(meas.uncert, [0, 0, 0, 0, 0])
                else:
                    np.testing.assert_array_equal(meas.value, [5, 6, 7, 8, 9])
                    self.assertEqual(str(meas.unit), 'mA')
                    np.testing.assert_array_equal(meas.uncert, [0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
