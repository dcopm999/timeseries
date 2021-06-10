
import unittest

from timeseries import TimeSeries


class CaseEMA(unittest.TestCase):
    def setUp(self):
        self.values = [292.2, 422.0, 671.3, 951.4, 1049.3, 1247.4, 2450.5, 3794.6, 4843.7, 6984.4, 9628.9, 12516.4, 15531.3, 15468.5, 18986.1, 23511.9, 30459.6, 43274.1]
        self.target = 52747.5
        self.amount = 0.00001
        self.ema = TimeSeries.EMA()
        self.alpha = 0.05870315789475225
        self.predict = 52748.331325599

    def test_fit(self):
        alpha = self.ema.fit(self.values, self.target, self.amount)
        self.assertEqual(alpha, self.alpha)

    def test_predict(self):
        predict = self.ema.predict(self.values, self.alpha)
        self.assertEqual(predict, self.predict)

    def test_mse(self):
        mse = self.ema.mse(self.target, self.predict)
        self.assertEqual(mse, 0.6911022515521822)
