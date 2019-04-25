import unittest
from prep import DrugEncoderV2, PrepareData


class TestDrugEncoderV2(unittest.TestCase):
    def test__init__(self):
        drug_enc = DrugEncoderV2(66, 77)
        self.assertEqual(drug_enc.duration_from_admission_in_hours, 66,
                         'duration_from_admission_in_hours not initialized')
        self.assertEqual(drug_enc.max_number_of_drug_features, 77,
                         'max_number_of_drug_features not initialized')

    def test_fit(self):
        pre = PrepareData('test', '../../data/demo_data/all')
        pre.read_table('prescriptions')
        drug_enc = DrugEncoderV2()
        drug_enc.fit(pre.prescriptions)
        self.assertEqual(drug_enc.drug_mapping['NS500'], int(1), 'Drug mapping not correct.')
        self.assertEqual(drug_enc.drug_mapping['MORPH100'], int(797), 'Drug mapping not correct.')

        # Test Encoder Parameters
        pre = PrepareData('test', '../../data/demo_data/all')
        pre.read_table('prescriptions')
        drug_enc = DrugEncoderV2(48, 13)
        drug_enc.fit(pre.prescriptions)
        self.assertEqual(len(drug_enc.drug_mapping), int(13), 'max_number_of_drug_features does not work.')

    def test__get_drug_feature(self):
        pre = PrepareData('test', '../../data/demo_data/all')
        pre.read_table('prescriptions')

        drug_enc = DrugEncoderV2( 48, None)
        drug_enc.fit(pre.prescriptions)
        unique_drugs = drug_enc._get_drug_feature(pre.prescriptions)
        self.assertEqual(len(unique_drugs),
                         len(set(unique_drugs)),
                         'Error in DrugEncocder._get_drug_feature(); list elements are not unique;')

    def test_transform(self):
        pre = PrepareData('test', '../../data/demo_data/all')
        pre.read_tables(['prescriptions', 'admissions'])

        drug_enc = DrugEncoderV2(48, 10)
        drug_enc.fit(pre.prescriptions)
        hadm_drug_matrix, hadm_id_mapping = drug_enc.transform(pre.prescriptions, pre.admissions)

        self.assertTrue(((hadm_drug_matrix[7, :] == [0, 1, 1, 1, 0, 1, 1, 1, 1, 0]) * 1).sum(),
                        'Error in DrugEncoder.transform(); hadm_drug_matrix wrong')

        self.assertEqual(hadm_drug_matrix.shape,
                         (len(pre.admissions), 10),
                         'Error in DrugEncoder.transform();  hadm_drug_matrix has wrong shape;')

        self.assertEqual(len(hadm_id_mapping),
                         pre.admissions['HADM_ID'].unique().shape[0],
                         'Error in DrugEncoder.transform(); wrong shape;')

        self.assertEqual(drug_enc.drug_mapping,
                         {'FURO40I': 0, 'NS500': 1, 'NS1000': 2, 'NACLFLUSH': 3, 'INSULIN': 4, 'D5W250': 5, 'VANC1F': 6,
                          'VANCOBASE': 7, 'HEPA5I': 8, 'KCL20PM': 9},
                         'Error in DrugEncoder.transform(); drug_mapping is wrong')
