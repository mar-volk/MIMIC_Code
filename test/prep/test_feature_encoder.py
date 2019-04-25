import unittest
from prep import FeatureEncoder

import pandas as pd


class TestFeatureEncoder(unittest.TestCase):
    def test_init(self):
        feat_enc = FeatureEncoder(column_name='NAME' ,
                                  min_frequency_of_feature=2,
                                  max_number_features=999
                                  )
        self.assertTrue(feat_enc.column_name == 'NAME')
        self.assertTrue(feat_enc.min_frequency_of_feature == 2)
        self.assertTrue(feat_enc.max_number_features == 999)

    def test_fit(self):
        table = pd.DataFrame()
        table['HADM_ID'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        table['SEX'] = ['M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'NA']

        # Test limitation by min_frequency_of_feature
        feat_enc = FeatureEncoder(column_name='SEX',
                                  min_frequency_of_feature=3,
                                  max_number_features=999)
        feat_enc.fit(table)
        self.assertTrue(feat_enc.feature_mapping['M'] == 0)
        self.assertTrue(feat_enc.feature_mapping['F'] == 1)
        self.assertTrue(len(feat_enc.feature_mapping) == 2)

        # Test limitation by min_frequency_of_feature
        feat_enc = FeatureEncoder(column_name='SEX',
                                  min_frequency_of_feature=0,
                                  max_number_features=1)
        feat_enc.fit(table)
        self.assertTrue(feat_enc.feature_mapping['M'] == 0)
        self.assertTrue(len(feat_enc.feature_mapping) == 1)

        # Test without limitation
        feat_enc = FeatureEncoder(column_name='SEX',
                                  min_frequency_of_feature=0,
                                  max_number_features=99999)
        feat_enc.fit(table)
        self.assertTrue(feat_enc.feature_mapping['M'] == 0)
        self.assertTrue(len(feat_enc.feature_mapping) == 3)

    def test_transform(self):
        table = pd.DataFrame()
        table['HADM_ID'] = [1, 2222, 33, 44, 55, 66, 777, 8888, 99, 101010]
        table['SEX'] = ['M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'NA']

        hadm_id_mapping = {}
        for i in range(0, len(table['HADM_ID'])):
            hadm_id_mapping[table['HADM_ID'][i]] = i
        feat_enc = FeatureEncoder(column_name='SEX',
                                  min_frequency_of_feature=0,
                                  max_number_features=999)
        feat_enc.fit(table)
        x = feat_enc.transform(table, hadm_id_mapping)

        self.assertEqual(((x == [[1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]]) * 1).sum(), 30)
