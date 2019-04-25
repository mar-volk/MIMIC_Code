from .drug_encoder_v2 import DrugEncoderV2
from .feature_encoder import FeatureEncoder
import prep.utils as utils
import numpy as np


class PatientEncoder:
    def __init__(self, *,
                 duration_from_admission_in_hours=48,

                 max_number_of_drug_features=None,
                 # num_drug_features=2500,

                 max_num_of_diagnoses=1000,
                 min_frequency_of_diag=3,

                 include_age_at_admission=True,

                 max_number_admission_type_features=100,
                 min_frq_admission_type_features=3,

                 max_number_admission_location_features=100,
                 min_frq_admission_location_features=3,

                 max_number_insurance_features=100,
                 min_frq_insurance_features=3,

                 min_frq_marital_features=100,
                 max_number_marital_features=3):

        self.duration_from_admission_in_hours = duration_from_admission_in_hours
        self.max_number_of_drug_features = max_number_of_drug_features
        # self.num_drug_features = num_drug_features
        self.max_num_of_diagnoses = max_num_of_diagnoses
        self.min_frequency_of_diag = min_frequency_of_diag
        self.include_age_at_admission = include_age_at_admission

        self.max_number_admission_type_features = max_number_admission_type_features
        self.min_frq_admission_type_features = min_frq_admission_type_features

        self.max_number_admission_location_features = max_number_admission_location_features
        self.min_frq_admission_location_features = min_frq_admission_location_features

        self.max_number_insurance_features = max_number_insurance_features
        self.min_frq_insurance_features = min_frq_insurance_features

        self.max_number_marital_features = max_number_marital_features
        self.min_frq_marital_features = min_frq_marital_features

        self.drug_enc = DrugEncoderV2(duration_from_admission_in_hours=self.duration_from_admission_in_hours,
                                    max_number_of_drug_features=self.max_number_of_drug_features)

        self.diag_enc = FeatureEncoder('DIAGNOSIS',
                                       self.min_frequency_of_diag,
                                       self.max_num_of_diagnoses)

        self.adm_type_enc = FeatureEncoder('ADMISSION_TYPE',
                                           self.min_frq_admission_type_features,
                                           self.max_number_admission_type_features)

        self.adm_loc_enc = FeatureEncoder('ADMISSION_LOCATION',
                                          self.min_frq_admission_location_features,
                                          self.max_number_admission_location_features)

        self.insur_enc = FeatureEncoder('INSURANCE',
                                        self.min_frq_insurance_features,
                                        self.max_number_insurance_features)

        self.marital_enc = FeatureEncoder('MARITAL_STATUS',
                                          self.min_frq_marital_features,
                                          self.max_number_marital_features)

    def fit(self,
            prescriptions,
            admissions,
            patients):

        self.drug_enc.fit(prescriptions)
        self.diag_enc.fit(admissions)
        self.adm_type_enc.fit(admissions)
        self.adm_loc_enc.fit(admissions)
        self.insur_enc.fit(admissions)
        self.marital_enc.fit(admissions)

    def transform(self, prescriptions, admissions, patients):
        drug, hadm_id_map = self.drug_enc.transform(prescriptions, admissions)
        diag = self.diag_enc.transform(admissions, hadm_id_map)

        age = utils.get_age_at_admission(admissions, patients, hadm_id_map) / 100  # For normalization
        adm_type = self.adm_type_enc.transform(admissions, hadm_id_map)
        adm_loc = self.adm_loc_enc.transform(admissions, hadm_id_map)
        insur = self.insur_enc.transform(admissions, hadm_id_map)
        marital = self.marital_enc.transform(admissions, hadm_id_map)

        x = np.concatenate((drug,
                            diag,
                            adm_type,
                            adm_loc,
                            insur,
                            marital,
                            age,
                            ),
                            axis=1)
        return x, hadm_id_map
