import pandas as pd
import numpy as np


class DrugEncoder:

    def __init__(self, prescriptions, duration_from_admission_in_hours=48, max_number_of_drug_features=None):
        self.duration_from_admission_in_hours = duration_from_admission_in_hours
        self.max_number_of_drug_features = max_number_of_drug_features
        self.drug_features = self._get_drug_feature(prescriptions)
        self.drug_mapping = self._get_drug_mapping()

    def _get_drug_feature(self, prescriptions):
        # prepare list of unique drugs
        unique_drugs_table = prescriptions.groupby('FORMULARY_DRUG_CD').count().sort_values(by='HADM_ID', ascending=False)
        unique_drugs = unique_drugs_table.index.values
        self.unique_drugs = unique_drugs
        self.unique_drugs_frequency = unique_drugs_table['SUBJECT_ID'].values

        if self.max_number_of_drug_features is None:
            number_drug_features = len(unique_drugs)
        else:
            number_drug_features = min(self.max_number_of_drug_features, len(unique_drugs))
        drug_features = unique_drugs[0:number_drug_features]
        return drug_features

    def _get_drug_mapping(self):
        drug_mapping = {}
        for i in range(0, len(self.drug_features)):
            drug_mapping[self.drug_features[i]] = i
        return drug_mapping

    def transform(self, prescriptions, admissions):
        # returns: - a one-hot-encoded matrix of prescribed drugs
        #          - hadm_id - mapping
        #            (drug_mapping from instance variable)

        adm_pres = admissions.merge(prescriptions, left_on='HADM_ID', right_on='HADM_ID')

        # we only observe prescribed drugs from first x hours
        observation_time = pd.Timedelta(hours=self.duration_from_admission_in_hours)

        out_of_obs = adm_pres['STARTDATE'] - adm_pres['ADMITTIME'] > observation_time
        drop_indices = np.where(out_of_obs)[0]
        adm_pres = adm_pres.drop(index=drop_indices).reset_index()

        # # drop prescriptions without GSN-Code
        # nan_rows = np.where(adm_pres['GSN'].isna())[0]
        # adm_pres = adm_pres.drop(index=nan_rows).reset_index()

        # hadm_id mapping
        hadm_id_unique = admissions['HADM_ID'].unique()
        hadm_id_mapping = {}
        for i in range(0, len(hadm_id_unique)):
            hadm_id_mapping[hadm_id_unique[i]] = i

        # prepare matrix
        hadm_drug_matrix = np.zeros((len(hadm_id_mapping), len(self.drug_mapping)), dtype=np.int8)
        for i in range(0, len(adm_pres)):
            drug = adm_pres['FORMULARY_DRUG_CD'][i]
            hadm_id = adm_pres['HADM_ID'][i]

            if drug in self.drug_mapping:
                row = hadm_id_mapping[hadm_id]
                col = self.drug_mapping[drug]
                hadm_drug_matrix[row, col] = 1

        return hadm_drug_matrix, hadm_id_mapping
