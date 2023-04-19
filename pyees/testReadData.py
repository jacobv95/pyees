import unittest
import numpy as np
try:
    from readData import sheetsFromFile
except ImportError:
    from pyees.readData import sheetsFromFile

class test(unittest.TestCase):

    def testReadFileTypes(self):
        # xlsx file
        dat = sheetsFromFile('testData/data1.xlsx', 'A-B')
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.b.unit), 'mA')
        np.testing.assert_array_equal(dat.b.uncert, [0, 0, 0, 0, 0])

        # xls file
        dat = sheetsFromFile('testData/data1.xls', 'A-B')
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.b.unit), 'mA')
        np.testing.assert_array_equal(dat.b.uncert, [0, 0, 0, 0, 0])

        # csv file
        with self.assertRaises(Exception) as context:
            dat = sheetsFromFile('testData/data1.csv', 'A-B')
        self.assertTrue("The file extension is not supported. The supported extension are ['.xls', '.xlsx']" in str(context.exception))

    def testReadUncertanty(self):
        dat = sheetsFromFile('testData/data3.xlsx', 'A-B', 'C-D')
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'm')
        np.testing.assert_array_equal(dat.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.b.unit), 'mA')
        np.testing.assert_array_equal(dat.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])

        dat = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_almost_equal([elemA.covariance[elemB] for elemA, elemB in zip(dat.a, dat.b)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        np.testing.assert_array_equal(dat.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.b.unit), 'mA')
        np.testing.assert_array_equal(dat.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elemB.covariance[elemA] for elemA, elemB in zip(dat.a, dat.b)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)

        with self.assertRaises(Exception) as context:
            sheetsFromFile('testData/data6.xlsx', 'A-B', 'C-D')
        self.assertTrue("The covariances has to be symmetric" in str(context.exception))

        dat = sheetsFromFile('testData/data5.xlsx', dataRange="B-C", uncertRange="H-I")
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_almost_equal([elemA.covariance[elemC] for elemA, elemC in zip(dat.a, dat.c)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        np.testing.assert_array_equal(dat.c.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.c.unit), 'mA')
        np.testing.assert_array_equal(dat.c.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elemC.covariance[elemA] for elemA, elemC in zip(dat.a, dat.c)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        
        dat = sheetsFromFile('testData/data5.xlsx', dataRange="B-C", uncertRange="K-L")
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        for elemA, elemC in zip(dat.a, dat.c):
            self.assertFalse(elemC in elemA.covariance)
        np.testing.assert_array_equal(dat.c.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.c.unit), 'mA')
        np.testing.assert_array_equal(dat.c.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])
        for elemA, elemC in zip(dat.a, dat.c):
            self.assertFalse(elemA in elemC.covariance)
        

    def testAppend(self):
        dat1 = sheetsFromFile('testData/data1.xlsx', 'A-B')
        dat2 = sheetsFromFile('testData/data2.xlsx', 'A-B')
        dat1.append(dat2)
        np.testing.assert_array_equal(dat1.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat1.a.uncert, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat1.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat1.b.unit), 'mA')
        np.testing.assert_array_equal(dat1.b.uncert, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        dat4 = sheetsFromFile('testData/data4.xlsx', 'A-B')
        dat5 = sheetsFromFile('testData/data5.xlsx', 'B-C')
    
        with self.assertRaises(Exception) as context:
            dat4.append(dat5)
        self.assertTrue("You can only append sheets with the excact same measurements. The names did not match" in str(context.exception))

        dat3 = sheetsFromFile('testData/data3.xlsx', 'A-B')
        dat4 = sheetsFromFile('testData/data4.xlsx', 'A-B')
        with self.assertRaises(Exception) as context:
            dat3.append(dat4)
        self.assertTrue("You can not set an element of [1, 2, 3, 4, 5] [m] with [1, 2, 3, 4, 5] [L/min] as they do not have the same unit" in str(context.exception))

        dat2 = sheetsFromFile('testData/data2.xlsx', 'A-B', 'C-D')
        dat4 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat2.append(dat4)
        np.testing.assert_array_equal(dat2.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat2.a.unit), 'L/min')
        np.testing.assert_array_equal(dat2.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat2.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat2.b.unit), 'mA')
        np.testing.assert_array_equal(dat2.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9])
        cov = np.array([0, 0, 0, 0, 0, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 1000
        for i, (elemA, elemB) in enumerate(zip(dat2.a, dat2.b)):
            if elemB in elemA.covariance:
                self.assertAlmostEqual(elemA.covariance[elemB], cov[i])
                self.assertAlmostEqual(elemB.covariance[elemA], cov[i])
                
            
        dat2 = sheetsFromFile('testData/data2.xlsx', 'A-B', 'C-D')
        dat4 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat4.append(dat2)
        np.testing.assert_array_equal(dat4.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat4.a.unit), 'L/min')
        np.testing.assert_array_equal(dat4.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat4.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat4.b.unit), 'mA')
        np.testing.assert_array_equal(dat4.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9])
        cov = np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0, 0, 0, 0, 0]) / 1000 / 60 / 1000
        for i, (elemA, elemB) in enumerate(zip(dat4.a, dat4.b)):
            if elemB in elemA.covariance:
                self.assertAlmostEqual(elemA.covariance[elemB], cov[i])
                self.assertAlmostEqual(elemB.covariance[elemA], cov[i])
       
        dat4_1 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat4_2 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat4_1.append(dat4_2)
        np.testing.assert_array_equal(dat4_1.a.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat4_1.a.unit), 'L/min')
        np.testing.assert_array_equal(dat4_1.a.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat4_1.b.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat4_1.b.unit), 'mA')
        np.testing.assert_array_equal(dat4_1.b.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9])
        np.testing.assert_almost_equal([elemA.covariance[elemB] for elemA, elemB in zip(dat4_1.a, dat4_1.b)], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 6000)
        np.testing.assert_almost_equal([elemB.covariance[elemA] for elemA, elemB in zip(dat4_1.a, dat4_1.b)], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 6000)

    def testIndex(self):
        dat = sheetsFromFile('testData/data1.xlsx', 'A-B')
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.b.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.b.unit), 'mA')
        np.testing.assert_array_equal(dat.b.uncert, [0, 0, 0, 0, 0])

        dat = dat[0:3]
        np.testing.assert_array_equal(dat.a.value, [1, 2, 3])
        self.assertEqual(str(dat.a.unit), 'L/min')
        np.testing.assert_array_equal(dat.a.uncert, [0, 0, 0])
        np.testing.assert_array_equal(dat.b.value, [5, 6, 7])
        self.assertEqual(str(dat.b.unit), 'mA')
        np.testing.assert_array_equal(dat.b.uncert, [0, 0, 0])

    def testIterable(self):
        # xlsx file
        dat = sheetsFromFile('testData/data1.xlsx', 'A-B')
        for i, meas in enumerate(dat):
            if i == 0:
                np.testing.assert_array_equal(meas.value, [1, 2, 3, 4, 5])
                self.assertEqual(str(meas.unit), 'L/min')
                np.testing.assert_array_equal(meas.uncert, [0, 0, 0, 0, 0])
            else:
                np.testing.assert_array_equal(meas.value, [5, 6, 7, 8, 9])
                self.assertEqual(str(meas.unit), 'mA')
                np.testing.assert_array_equal(meas.uncert, [0, 0, 0, 0, 0])

    def testNontypeInput(self):
        dat = sheetsFromFile("testData/data7.xlsx", "A-F")
        for sheet in dat:
            for elem in sheet:
                self.assertEqual(elem.unit, '1')

    def testsSheets(self):
        sheet = sheetsFromFile('testData/data8.xlsx', "A-B", sheets=0)
        self.assertTrue(hasattr(sheet, 'a'))
        self.assertTrue(hasattr(sheet, 'b'))
        np.testing.assert_array_equal(sheet.a.value, [1,2,3])
        self.assertEqual(sheet.a.unit, 'm')
        np.testing.assert_array_equal(sheet.b.value, [4,5,6])
        self.assertEqual(sheet.b.unit, 's')
        
        sheet = sheetsFromFile('testData/data8.xlsx', "A-C", sheets=1)
        self.assertTrue(hasattr(sheet, 'c'))
        self.assertTrue(hasattr(sheet, 'd'))
        self.assertTrue(hasattr(sheet, 'e'))
        np.testing.assert_array_equal(sheet.c.value, [7,8,9,10,7,8,9,10])
        self.assertEqual(sheet.c.unit, 'Hz')
        np.testing.assert_array_equal(sheet.d.value, [11,12,13,14,11,12,13,14])
        self.assertEqual(sheet.d.unit, 'L')
        np.testing.assert_array_equal(sheet.e.value, [15,16,17,18,15,16,17,18])
        self.assertEqual(sheet.e.unit, 'C')
        
        sheets = sheetsFromFile('testData/data8.xlsx', "A-B", sheets=[0,1])
        self.assertTrue(hasattr(sheets[0], 'a'))
        self.assertTrue(hasattr(sheets[0], 'b'))
        np.testing.assert_array_equal(sheets[0].a.value, [1,2,3])
        self.assertEqual(sheets[0].a.unit, 'm')
        np.testing.assert_array_equal(sheets[0].b.value, [4,5,6])
        self.assertEqual(sheets[0].b.unit, 's')
        self.assertTrue(hasattr(sheets[1], 'c'))
        self.assertTrue(hasattr(sheets[1], 'd'))
        self.assertFalse(hasattr(sheets[1], 'e'))
        np.testing.assert_array_equal(sheets[1].c.value, [7,8,9,10,7,8,9,10])
        self.assertEqual(sheets[1].c.unit, 'Hz')
        np.testing.assert_array_equal(sheets[1].d.value, [11,12,13,14,11,12,13,14])
        self.assertEqual(sheets[1].d.unit, 'L')
        
        sheets = sheetsFromFile('testData/data8.xlsx', ["A-B", "A-C"], sheets=[0,1])
        self.assertTrue(hasattr(sheets[0], 'a'))
        self.assertTrue(hasattr(sheets[0], 'b'))
        np.testing.assert_array_equal(sheets[0].a.value, [1,2,3])
        self.assertEqual(sheets[0].a.unit, 'm')
        np.testing.assert_array_equal(sheets[0].b.value, [4,5,6])
        self.assertEqual(sheets[0].b.unit, 's')
        self.assertTrue(hasattr(sheets[1], 'c'))
        self.assertTrue(hasattr(sheets[1], 'd'))
        self.assertTrue(hasattr(sheets[1], 'e'))
        np.testing.assert_array_equal(sheets[1].c.value, [7,8,9,10,7,8,9,10])
        self.assertEqual(sheets[1].c.unit, 'Hz')
        np.testing.assert_array_equal(sheets[1].d.value, [11,12,13,14,11,12,13,14])
        self.assertEqual(sheets[1].d.unit, 'L')
        np.testing.assert_array_equal(sheets[1].e.value, [15,16,17,18,15,16,17,18])
        self.assertEqual(sheets[1].e.unit, 'C')
        
        
        
        sheets = sheetsFromFile('testData/data8.xlsx', dataRange=["A-B", "A-C"], uncertRange=["E-F","E-G"], sheets=[0,1])
        self.assertTrue(hasattr(sheets[0], 'a'))
        self.assertTrue(hasattr(sheets[0], 'b'))
        np.testing.assert_array_equal(sheets[0].a.value, [1,2,3])
        np.testing.assert_array_equal(sheets[0].a.uncert, np.array([1,2,3]) / 100)
        self.assertEqual(sheets[0].a.unit, 'm')
        np.testing.assert_array_equal(sheets[0].b.value, [4,5,6])
        np.testing.assert_array_equal(sheets[0].b.uncert, np.array([4,5,6]) / 100)
        self.assertEqual(sheets[0].b.unit, 's')
        self.assertTrue(hasattr(sheets[1], 'c'))
        self.assertTrue(hasattr(sheets[1], 'd'))
        self.assertTrue(hasattr(sheets[1], 'e'))
        np.testing.assert_array_equal(sheets[1].c.value, [7,8,9,10,7,8,9,10])
        np.testing.assert_array_equal(sheets[1].c.uncert,  np.array([7,8,9,10,7,8,9,10])/100)
        self.assertEqual(sheets[1].c.unit, 'Hz')
        np.testing.assert_array_equal(sheets[1].d.value, [11,12,13,14,11,12,13,14])
        np.testing.assert_array_equal(sheets[1].d.uncert,  np.array([11,12,13,14,11,12,13,14])/100)
        self.assertEqual(sheets[1].d.unit, 'L')
        np.testing.assert_array_equal(sheets[1].e.value, [15,16,17,18,15,16,17,18])
        np.testing.assert_array_equal(sheets[1].e.uncert, np.array([15,16,17,18,15,16,17,18])/100)
        self.assertEqual(sheets[1].e.unit, 'C')
        
        
        
        
if __name__ == '__main__':
    unittest.main()
