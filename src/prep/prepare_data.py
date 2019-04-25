import pandas as pd
import pickle
import glob
import os


class PrepareData:

    def __init__(self, name='name', root='../data/train'):#

        self.name = name
        self.root = root
        self.root_pickle = self.root + '/pickle/'
        self.root_raw = self.root + '/raw/'

        # indicators for timedate
        # so far only for: patients, admission, prescriptions, cptevents, callout
        self.column_time_indicator = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME', 'DOB', 'DOD',
                                      'DOD_HOSP', 'DOD_SSN', 'CHARTDATE', 'STARTDATE', 'ENDDATE', 'CREATETIME',
                                      'UPDATETIME', 'ACKNOWLEDGETIME', 'OUTCOMETIME', 'FIRSTRESERVATIONTIME',
                                      'CURRENTRESERVATIONTIME']

    def read_table(self, table_name, store_pickle=False):
        # read data from csv

        file_name_csv = self.root_raw + table_name.upper() + '.csv'
        table = pd.read_csv(file_name_csv)

        # set datetime-columns
        cols = table.columns.values.tolist()
        time_cols = []
        for col in cols:
            if col in self.column_time_indicator:
                time_cols.append(col)

        for col in time_cols:
            table[col] = table[col].apply(lambda x: pd.to_datetime(x))

        exec('self.' + table_name.lower() + ' = table')

        # store pickle-file
        if store_pickle:
            file_name_pickle = self.root_pickle + table_name.lower() +'.p'
            pickle.dump(table, open(file_name_pickle, "wb"))
        return table

    def read_tables(self, table_name_list, store_pickle=False):
        for table_name in table_name_list:
            self.read_table(table_name, store_pickle)

    def load_table_pickle(self, table_name):
        file_name_pickle = self.root_pickle + table_name.lower() + '.p'
        table = pickle.load(open(file_name_pickle, 'rb'))
        exec('self.' + table_name.lower() + ' = table')
        return table

    def load_table_pickle_list(self, table_name_list):
        for table_name in table_name_list:
            self.load_table_pickle(table_name)

    def get_table_names(self, lower_case=True):
        list_of_files = glob.glob(self.root_raw + '*.csv')
        table_names = []
        for path in list_of_files:
            if lower_case:
                table_name = os.path.basename(path)[0:-4].lower()
            else:
                table_name = os.path.basename(path)[0:-4].upper()
            table_names.append(table_name)
        return table_names

