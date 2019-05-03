import pickle
from prepare_data_v2 import PrepareDataV2


pre = PrepareDataV2('all', '../../data/all')
table_name_list = ['admissions', 'patients', 'prescriptions']
pre.read_tables(table_name_list, store_pickle=True)
print('tables read')

# Split the admissions table ( 20% / 20% / 60% )
admissions_test = pre.admissions[pre.admissions['SUBJECT_ID'] % 5 == 0]
admissions_val = pre.admissions[pre.admissions['SUBJECT_ID'] % 5 == 1]
admissions_train = pre.admissions[pre.admissions['SUBJECT_ID'] % 5 > 1]

# Check for overlap
test_id = admissions_test['SUBJECT_ID']
train_id = admissions_train['SUBJECT_ID']
val_id = admissions_val['SUBJECT_ID']
overlap2 = len(admissions_val.merge(admissions_test, on='SUBJECT_ID'))
overlap1 = len(admissions_test.merge(admissions_train, on='SUBJECT_ID'))
overlap3 = len(admissions_val.merge(admissions_train, on='SUBJECT_ID'))
assert overlap1 + overlap2 + overlap3 == 0

# save files with pickle
folder_train = '../../data/train/pickle/admissions.p'
folder_test = '../../data/test/pickle/admissions.p'
folder_val = '../../data/val/pickle/admissions.p'
pickle.dump(admissions_train, open(folder_train, "wb"))
pickle.dump(admissions_test, open(folder_test, "wb"))
pickle.dump(admissions_val, open(folder_val, "wb"))

# Split the prescriptions table (20% / 20% / 60%)
prescriptions_test = pre.prescriptions[pre.prescriptions['SUBJECT_ID'] % 5 == 0]
prescriptions_val = pre.prescriptions[pre.prescriptions['SUBJECT_ID'] % 5 == 1]
prescriptions_train = pre.prescriptions[pre.prescriptions['SUBJECT_ID'] % 5 > 1]

# Check for overlap
test_id = prescriptions_test['SUBJECT_ID']
train_id = prescriptions_train['SUBJECT_ID']
val_id = prescriptions_val['SUBJECT_ID']
overlap1 = len(prescriptions_test.merge(prescriptions_train, on='SUBJECT_ID'))
overlap2 = len(prescriptions_val.merge(prescriptions_test, on='SUBJECT_ID'))
overlap3 = len(prescriptions_val.merge(prescriptions_train, on='SUBJECT_ID'))
assert overlap1 + overlap2 + overlap3 == 0

# save files with pickle
folder_train = '../../data/train/pickle/prescriptions.p'
folder_test = '../../data/test/pickle/prescriptions.p'
folder_val = '../../data/val/pickle/prescriptions.p'
pickle.dump(prescriptions_train, open(folder_train, "wb"))
pickle.dump(prescriptions_test, open(folder_test, "wb"))
pickle.dump(prescriptions_val, open(folder_val, "wb"))

# Split the patients table (20% / 20% / 60%)
patients_test = pre.patients[pre.patients['SUBJECT_ID'] % 5 == 0]
patients_val = pre.patients[pre.patients['SUBJECT_ID'] % 5 == 1]
patients_train = pre.patients[pre.patients['SUBJECT_ID'] % 5 > 1]

# Check for overlap
test_id = patients_test['SUBJECT_ID']
train_id = patients_train['SUBJECT_ID']
val_id = patients_val['SUBJECT_ID']
overlap1 = len(patients_test.merge(patients_train, on='SUBJECT_ID'))
overlap2 = len(patients_val.merge(patients_test, on='SUBJECT_ID'))
overlap3 = len(patients_val.merge(patients_train, on='SUBJECT_ID'))
assert overlap1 + overlap2 + overlap3 == 0

# save files with pickle
folder_train = '../../data/train/pickle/patients.p'
folder_test = '../../data/test/pickle/patients.p'
folder_val = '../../data/val/pickle/patients.p'
pickle.dump(patients_train, open(folder_train, "wb"))
pickle.dump(patients_test, open(folder_test, "wb"))
pickle.dump(patients_val, open(folder_val, "wb"))
