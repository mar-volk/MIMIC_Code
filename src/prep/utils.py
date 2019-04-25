import pandas as pd
import numpy as np


def patient_alive_after_duration(admissions, patients, hadm_id_mapping,
                                 alive_duration_in_days=14):
    adm_pat = admissions.merge(patients, on='SUBJECT_ID')
    alive_duration = pd.Timedelta(days=alive_duration_in_days)
    duration = adm_pat['DOD'] - adm_pat['ADMITTIME']
    still_alive = duration.isnull()
    alive_longer_than_dur = duration >= alive_duration
    adm_pat['ALIVE_AFTER_DUR'] = (still_alive | alive_longer_than_dur) * 1

    hadm_id_list = list(hadm_id_mapping.keys())

    alive_after_dur_array = np.zeros(len(hadm_id_mapping))
    for i in range(0, len(hadm_id_mapping)):
        hadm_id = hadm_id_list[i]
        alive_after_dur = adm_pat.loc[adm_pat['HADM_ID'] == hadm_id]['ALIVE_AFTER_DUR']
        alive_after_dur_array[i] = alive_after_dur
    return alive_after_dur_array


def get_duration_of_stay_in_days(admissions, hadm_id_mapping):
    time1 = admissions['DISCHTIME'].values.astype(int) / int(1e9)
    time2 = admissions['ADMITTIME'].values.astype(int) / int(1e9)
    duration_in_days = (time1 - time2).astype(float) / 60 / 60 / 24

    duration_of_stay_in_days_hadm_id_mapping = np.zeros(len(hadm_id_mapping))
    for i in range(0, len(duration_in_days)):
        raw = raw = hadm_id_mapping[admissions.iloc[i]['HADM_ID']]
        duration_of_stay_in_days_hadm_id_mapping[raw] = duration_in_days[i]
    return duration_of_stay_in_days_hadm_id_mapping


def get_age_at_admission(admissions, patients, hadm_id_mapping):
    adm_pat = admissions.merge(patients, on='SUBJECT_ID')

    # This is a work-arround because pandas cannot calculate easily with these dates that are seperated over several
    # hundred years. (base unit = 1 ns, 64 bis)
    time1 = adm_pat['ADMITTIME'].values.astype(int) / int(1e9)
    time2 = adm_pat['DOB'].values.astype(int) / int(1e9)
    age_at_adm_in_years = (time1 - time2) / 60 / 60 / 24 / 365.25

    # For de-anonymization the date of birth of old patients (>89years at first admission)
    # was set to 300 years before the date of first admission.
    # To get a plausible age the patients age will be set to 91.4 years at first admission.
    # This is the median age of patients, whose age was shifted to 300 years.

    age_at_adm_in_years_corrected = age_at_adm_in_years.copy()
    c = 300 - 91.4
    for i in range(0, len(age_at_adm_in_years)):
        if age_at_adm_in_years[i] > 200:
            age_at_adm_in_years_corrected[i] = age_at_adm_in_years[i] - c

    adm_pat['age_at_adm'] = age_at_adm_in_years_corrected

    age_at_adm = np.ones((len(hadm_id_mapping), 1), dtype=float) * np.nan
    for i in range(0, len(adm_pat)):
        raw = hadm_id_mapping[adm_pat['HADM_ID'][i]]
        age = age_at_adm_in_years_corrected[i]
        age_at_adm[raw] = age
    return age_at_adm


def get_binned_feature(admissions, patients, hadm_id_mapping, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]):
    age_at_adm = get_age_at_admission(admissions, patients, hadm_id_mapping)
    binned_age = np.digitize(age_at_adm, bins, right=False)
    return binned_age


