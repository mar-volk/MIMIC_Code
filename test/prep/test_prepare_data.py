import unittest
import pandas as pd
import pickle

from prep import PrepareData



class TestPrepareData(unittest.TestCase):
    def test_read_table(self):
        pre = PrepareData('test', '../../data/demo_data/all')
        admissions = pre.read_table('admissions')

        # Check data type of table
        self.assertEqual(type(admissions), pd.core.frame.DataFrame, 'Read table is not of type Pandas.DataFrame.')

        # Check if there is data in the table
        self.assertEqual(admissions['DIAGNOSIS'][2], 'SEPSIS', 'Data not read correctly')

        # Check if data is loaded correctly and if datetimes are marked correcly in table
        self.assertEqual((admissions['DISCHTIME'] - admissions['ADMITTIME'])[3],
                          pd.Timedelta('8 days 01:23:00'),
                          'Timedates not correctly marked in table.')

        # Test pickling files: First create empty file, use save-function of read_table, check if stored correctly
        file_name_pickle = '../../data/demo_data/all/pickle/admissions.p'
        empty_table = pd.DataFrame()
        pickle.dump(empty_table, open(file_name_pickle, "wb"))
        admissions = pre.read_table('admissions', True)
        loaded_table = pickle.load(open(file_name_pickle, 'rb'))
        print(loaded_table)
        self.assertEqual(loaded_table['DIAGNOSIS'][2], 'SEPSIS', 'Data is not correctly pickled.')

    def test_read_tables(self):
        pre = PrepareData('test', '../../data/demo_data/all')
        table_list = ['admissions', 'patients']
        pre.read_tables(table_list)

        # Check if the two tables are loaded
        self.assertEqual(pre.admissions['DIAGNOSIS'][2], 'SEPSIS', 'Not read multiple tables correctly.')
        self.assertEqual(pre.patients['SUBJECT_ID'][3], int(10017), 'Not read multiple tables correctly.')

    def test_load_table_pickle(self):
        pre_store = PrepareData('test1', '../../data/demo_data/all')
        pre_store.read_table('admissions', True)
        pre_load = PrepareData('test2', '../../data/demo_data/all')
        pre_load.load_table_pickle('admissions')
        self.assertEqual(pre_load.admissions['DIAGNOSIS'][2], 'SEPSIS', 'Not loaded table from pickle file.')

    def test_load_table_pickle_list(self):
        table_list = ['admissions', 'patients']
        pre_store = PrepareData('test1', '../../data/demo_data/all')
        pre_store.read_tables(table_list, True)
        pre_load = PrepareData('test2', '../../data/demo_data/all')
        pre_load.load_table_pickle_list(table_list)
        self.assertEqual(pre_load.admissions['DIAGNOSIS'][2], 'SEPSIS', 'Not loaded multipe tables from pickle files.')
        self.assertEqual(pre_load.patients['SUBJECT_ID'][3], int(10017), 'Not loaded multipe tables from pickle files.')

    def test_get_table_names(self):
        pre = PrepareData('test', '../../data/demo_data/all')
        tables_in_folder = pre.get_table_names(True)
        self.assertEqual(tables_in_folder[0], 'admissions', 'Does not return tables names correctly.')
        tables_in_folder = pre.get_table_names(False)
        self.assertEqual(tables_in_folder[0], 'ADMISSIONS', 'Does not return tables names correctly.')