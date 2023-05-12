import unittest
import numpy as np
try:
    from sheet import sheetsFromFile
except ImportError:
    from pyees.sheet import sheetsFromFile

class test(unittest.TestCase):

    def testReadFileTypes(self):
        # xlsx file
        dat = sheetsFromFile('testData/data1.xlsx', 'A-B')
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.B.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.B.unit), 'mA')
        np.testing.assert_array_equal(dat.B.uncert, [0, 0, 0, 0, 0])

        # xls file
        dat = sheetsFromFile('testData/data1.xls', 'A-B')
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.B.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.B.unit), 'mA')
        np.testing.assert_array_equal(dat.B.uncert, [0, 0, 0, 0, 0])

        # csv file
        with self.assertRaises(Exception) as context:
            dat = sheetsFromFile('testData/data1.Csv', 'A-B')
        self.assertTrue("The file extension is not supported. The supported extension are ['.xls', '.xlsx']" in str(context.exception))

    def testUncertMatrixVscovarianceMatrix(self):
        dat1 = sheetsFromFile('testData/data2.xlsx', 'A-B', 'C-D')
        dat2 = sheetsFromFile('testData/data2.xlsx', 'A-B', 'G-H')
        c1 = dat1.A * dat1.B
        c2 = dat2.A * dat2.B
        np.testing.assert_array_equal(c1.value, c2.value)
        self.assertEqual(c1.unit, c2.unit)
        np.testing.assert_array_equal(c1.uncert, c2.uncert)
                
    def testReadUncertanty(self):        
        dat = sheetsFromFile('testData/data3.xlsx', 'A-B', 'C-D')
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'm')
        np.testing.assert_array_equal(dat.A.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat.B.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.B.unit), 'mA')
        np.testing.assert_array_equal(dat.B.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])

        dat = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, np.sqrt([0.05, 0.1, 0.15, 0.2, 0.25]))
        np.testing.assert_almost_equal([elemA.covariance[elemB] for elemA, elemB in zip(dat.A, dat.B)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        np.testing.assert_array_equal(dat.B.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.B.unit), 'mA')
        np.testing.assert_array_equal(dat.B.uncert, np.sqrt([0.5, 0.6, 0.7, 0.8, 0.9]))
        np.testing.assert_almost_equal([elemB.covariance[elemA] for elemA, elemB in zip(dat.A, dat.B)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)

        with self.assertRaises(Exception) as context:
            sheetsFromFile('testData/data6.xlsx', 'A-B', 'C-D')
        self.assertTrue("The covariances has to be symmetric" in str(context.exception))

        dat = sheetsFromFile('testData/data5.xlsx', dataRange="B-C", uncertRange="H-I")
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, np.sqrt([0.05, 0.1, 0.15, 0.2, 0.25]))
        np.testing.assert_almost_equal([elemA.covariance[elemC] for elemA, elemC in zip(dat.A, dat.C)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        np.testing.assert_array_equal(dat.C.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.C.unit), 'mA')
        np.testing.assert_array_equal(dat.C.uncert, np.sqrt([0.5, 0.6, 0.7, 0.8, 0.9]))
        np.testing.assert_almost_equal([elemC.covariance[elemA] for elemA, elemC in zip(dat.A, dat.C)], np.array([0.025, 0.06, 0.105, 0.16, 0.225])/1000 / 60 / 1000)
        
        dat = sheetsFromFile('testData/data5.xlsx', dataRange="B-C", uncertRange="K-L")
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, [0.05, 0.1, 0.15, 0.2, 0.25])
        for elemA, elemC in zip(dat.A, dat.C):
            self.assertFalse(elemC in elemA.covariance)
        np.testing.assert_array_equal(dat.C.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.C.unit), 'mA')
        np.testing.assert_array_equal(dat.C.uncert, [0.5, 0.6, 0.7, 0.8, 0.9])
        for elemA, elemC in zip(dat.A, dat.C):
            self.assertFalse(elemA in elemC.covariance)
        

    def testAppend(self):
        dat1 = sheetsFromFile('testData/data1.xlsx', 'A-B')
        dat2 = sheetsFromFile('testData/data2.xlsx', 'A-B')
        dat1.append(dat2)
        np.testing.assert_array_equal(dat1.A.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat1.A.unit), 'L/min')
        np.testing.assert_array_equal(dat1.A.uncert, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat1.B.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat1.B.unit), 'mA')
        np.testing.assert_array_equal(dat1.B.uncert, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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
        np.testing.assert_array_equal(dat2.A.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat2.A.unit), 'L/min')
        np.testing.assert_array_equal(dat2.A.uncert, [0.05, 0.1, 0.15, 0.2, 0.25, 0.05**(1/2), 0.1**(1/2), 0.15**(1/2), 0.2**(1/2), 0.25**(1/2)])
        np.testing.assert_array_equal(dat2.B.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat2.B.unit), 'mA')
        np.testing.assert_array_equal(dat2.B.uncert, [0.5, 0.6, 0.7, 0.8, 0.9, 0.5**(1/2), 0.6**(1/2), 0.7**(1/2), 0.8**(1/2), 0.9**(1/2)])
        cov = np.array([0, 0, 0, 0, 0, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 1000
        for i, (elemA, elemB) in enumerate(zip(dat2.A, dat2.B)):
            if elemB in elemA.covariance:
                self.assertAlmostEqual(elemA.covariance[elemB], cov[i])
                self.assertAlmostEqual(elemB.covariance[elemA], cov[i])
                
            
        dat2 = sheetsFromFile('testData/data2.xlsx', 'A-B', 'C-D')
        dat4 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat4.append(dat2)
        np.testing.assert_array_equal(dat4.A.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat4.A.unit), 'L/min')
        np.testing.assert_array_equal(dat4.A.uncert, [0.05**(1/2), 0.1**(1/2), 0.15**(1/2), 0.2**(1/2), 0.25**(1/2), 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(dat4.B.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat4.B.unit), 'mA')
        np.testing.assert_array_equal(dat4.B.uncert, [0.5**(1/2), 0.6**(1/2), 0.7**(1/2), 0.8**(1/2), 0.9**(1/2), 0.5, 0.6, 0.7, 0.8, 0.9])
        cov = np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0, 0, 0, 0, 0]) / 1000 / 60 / 1000
        for i, (elemA, elemB) in enumerate(zip(dat4.A, dat4.B)):
            if elemB in elemA.covariance:
                self.assertAlmostEqual(elemA.covariance[elemB], cov[i])
                self.assertAlmostEqual(elemB.covariance[elemA], cov[i])
       
        dat4_1 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat4_2 = sheetsFromFile('testData/data4.xlsx', 'A-B', 'C-D')
        dat4_1.append(dat4_2)
        np.testing.assert_array_equal(dat4_1.A.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        self.assertEqual(str(dat4_1.A.unit), 'L/min')
        np.testing.assert_array_equal(dat4_1.A.uncert, np.sqrt([0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25]))
        np.testing.assert_array_equal(dat4_1.B.value, [5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        self.assertEqual(str(dat4_1.B.unit), 'mA')
        np.testing.assert_array_equal(dat4_1.B.uncert, np.sqrt([0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9]))
        np.testing.assert_almost_equal([elemA.covariance[elemB] for elemA, elemB in zip(dat4_1.A, dat4_1.B)], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 6000)
        np.testing.assert_almost_equal([elemB.covariance[elemA] for elemA, elemB in zip(dat4_1.A, dat4_1.B)], np.array([0.025, 0.06, 0.105, 0.16, 0.225, 0.025, 0.06, 0.105, 0.16, 0.225]) / 1000 / 60 / 6000)

    def testIndex(self):
        dat = sheetsFromFile('testData/data1.xlsx', 'A-B')
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3, 4, 5])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(dat.B.value, [5, 6, 7, 8, 9])
        self.assertEqual(str(dat.B.unit), 'mA')
        np.testing.assert_array_equal(dat.B.uncert, [0, 0, 0, 0, 0])

        dat = dat[0:3]
        np.testing.assert_array_equal(dat.A.value, [1, 2, 3])
        self.assertEqual(str(dat.A.unit), 'L/min')
        np.testing.assert_array_equal(dat.A.uncert, [0, 0, 0])
        np.testing.assert_array_equal(dat.B.value, [5, 6, 7])
        self.assertEqual(str(dat.B.unit), 'mA')
        np.testing.assert_array_equal(dat.B.uncert, [0, 0, 0])

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
        self.assertTrue(hasattr(sheet, 'A'))
        self.assertTrue(hasattr(sheet, 'B'))
        np.testing.assert_array_equal(sheet.A.value, [1,2,3])
        self.assertEqual(sheet.A.unit, 'm')
        np.testing.assert_array_equal(sheet.B.value, [4,5,6])
        self.assertEqual(sheet.B.unit, 's')
        
        sheet = sheetsFromFile('testData/data8.xlsx', "A-C", sheets=1)
        self.assertTrue(hasattr(sheet, 'C'))
        self.assertTrue(hasattr(sheet, 'D'))
        self.assertTrue(hasattr(sheet, 'E'))
        np.testing.assert_array_equal(sheet.C.value, [7,8,9,10,7,8,9,10])
        self.assertEqual(sheet.C.unit, 'Hz')
        np.testing.assert_array_equal(sheet.D.value, [11,12,13,14,11,12,13,14])
        self.assertEqual(sheet.D.unit, 'L')
        np.testing.assert_array_equal(sheet.E.value, [15,16,17,18,15,16,17,18])
        self.assertEqual(sheet.E.unit, 'C')
        
        sheets = sheetsFromFile('testData/data8.xlsx', "A-B", sheets=[0,1])
        self.assertTrue(hasattr(sheets[0], 'A'))
        self.assertTrue(hasattr(sheets[0], 'B'))
        np.testing.assert_array_equal(sheets[0].A.value, [1,2,3])
        self.assertEqual(sheets[0].A.unit, 'm')
        np.testing.assert_array_equal(sheets[0].B.value, [4,5,6])
        self.assertEqual(sheets[0].B.unit, 's')
        self.assertTrue(hasattr(sheets[1], 'C'))
        self.assertTrue(hasattr(sheets[1], 'D'))
        self.assertFalse(hasattr(sheets[1], 'E'))
        np.testing.assert_array_equal(sheets[1].C.value, [7,8,9,10,7,8,9,10])
        self.assertEqual(sheets[1].C.unit, 'Hz')
        np.testing.assert_array_equal(sheets[1].D.value, [11,12,13,14,11,12,13,14])
        self.assertEqual(sheets[1].D.unit, 'L')
        
        sheets = sheetsFromFile('testData/data8.xlsx', ["A-B", "A-C"], sheets=[0,1])
        self.assertTrue(hasattr(sheets[0], 'A'))
        self.assertTrue(hasattr(sheets[0], 'B'))
        np.testing.assert_array_equal(sheets[0].A.value, [1,2,3])
        self.assertEqual(sheets[0].A.unit, 'm')
        np.testing.assert_array_equal(sheets[0].B.value, [4,5,6])
        self.assertEqual(sheets[0].B.unit, 's')
        self.assertTrue(hasattr(sheets[1], 'C'))
        self.assertTrue(hasattr(sheets[1], 'D'))
        self.assertTrue(hasattr(sheets[1], 'E'))
        np.testing.assert_array_equal(sheets[1].C.value, [7,8,9,10,7,8,9,10])
        self.assertEqual(sheets[1].C.unit, 'Hz')
        np.testing.assert_array_equal(sheets[1].D.value, [11,12,13,14,11,12,13,14])
        self.assertEqual(sheets[1].D.unit, 'L')
        np.testing.assert_array_equal(sheets[1].E.value, [15,16,17,18,15,16,17,18])
        self.assertEqual(sheets[1].E.unit, 'C')
        
        
        
        sheets = sheetsFromFile('testData/data8.xlsx', dataRange=["A-B", "A-C"], uncertRange=["E-F","E-G"], sheets=[0,1])
        self.assertTrue(hasattr(sheets[0], 'A'))
        self.assertTrue(hasattr(sheets[0], 'B'))
        np.testing.assert_array_equal(sheets[0].A.value, [1,2,3])
        np.testing.assert_array_equal(sheets[0].A.uncert, np.array([1,2,3]) / 100)
        self.assertEqual(sheets[0].A.unit, 'm')
        np.testing.assert_array_equal(sheets[0].B.value, [4,5,6])
        np.testing.assert_array_equal(sheets[0].B.uncert, np.array([4,5,6]) / 100)
        self.assertEqual(sheets[0].B.unit, 's')
        self.assertTrue(hasattr(sheets[1], 'C'))
        self.assertTrue(hasattr(sheets[1], 'D'))
        self.assertTrue(hasattr(sheets[1], 'E'))
        np.testing.assert_array_equal(sheets[1].C.value, [7,8,9,10,7,8,9,10])
        np.testing.assert_array_equal(sheets[1].C.uncert,  np.array([7,8,9,10,7,8,9,10])/100)
        self.assertEqual(sheets[1].C.unit, 'Hz')
        np.testing.assert_array_equal(sheets[1].D.value, [11,12,13,14,11,12,13,14])
        np.testing.assert_array_equal(sheets[1].D.uncert,  np.array([11,12,13,14,11,12,13,14])/100)
        self.assertEqual(sheets[1].D.unit, 'L')
        np.testing.assert_array_equal(sheets[1].E.value, [15,16,17,18,15,16,17,18])
        np.testing.assert_array_equal(sheets[1].E.uncert, np.array([15,16,17,18,15,16,17,18])/100)
        self.assertEqual(sheets[1].E.unit, 'C')
        
        
        
        
if __name__ == '__main__':
    unittest.main()
