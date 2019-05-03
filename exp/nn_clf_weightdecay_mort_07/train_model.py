import torch
import models
import train
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle

from prep import PrepareDataV2, PatientEncoder
import prep.utils as utils


dtype = torch.float
device = torch.device("cpu")   #("cpu")  #("cuda:0")

p_dropout = 0.2
result_path = 'results/'
data_path = '../../data/'

# load data
tables = ['admissions', 'patients', 'prescriptions']
data_train = PrepareDataV2(name='train', root='../../data/train')
data_train.load_table_pickle_list(tables)
data_val = PrepareDataV2(name='val', root='../../data/val')
data_val.load_table_pickle_list(tables)
print('Data loaded')

# Prepare Predictors
pat_enc = PatientEncoder(duration_from_admission_in_hours=24,
                 max_number_of_drug_features=2500,
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
                 max_number_marital_features=3)

pat_enc.fit(data_train.prescriptions,
            data_train.admissions,
            data_train.patients)

pickle.dump(pat_enc, open('encoder.p', "wb"))  # Save encoder

x_train, hadm_id_map_train = pat_enc.transform(data_train.prescriptions,
                                               data_train.admissions,
                                               data_train.patients)

x_val, hadm_id_map_val = pat_enc.transform(data_val.prescriptions,
                                               data_val.admissions,
                                               data_val.patients)
print('Predictors prepared')

# Prepare targets
y_train = utils.patient_alive_after_duration(data_train.admissions, data_train.patients, hadm_id_map_train)
y_train = y_train.astype(np.uint8)
y_val = utils.patient_alive_after_duration(data_val.admissions, data_val.patients, hadm_id_map_val)
y_val = y_val.astype(np.uint8)

del data_train, data_val  # free RAM
print('all data prepared')

# Test different weight decays
weight_decay_list = 10**np.linspace(-5, -2, 10)

roc_list = []
result_path_list = []

for weight_decay in weight_decay_list:
    print('weight_decay: ' + str(weight_decay))
    # initialize model
    model = models.NNClfDropout2('NN_Clf_mort', x_train.shape[1], 2, p_dropout, torch.float, device=device)

    result_path = 'results_' + str(weight_decay) + '_/'
    # make sure model parameters are in GPU memory
    # model.cuda()

    # run training
    print('run model')
    start = time.time()
    training = train.TrainNN(model, result_path, x_train, y_train, x_val, y_val)
    training.run_train(100, lr=0.001, batch_size=64, weight_decay=weight_decay)

    print('time [s]:')
    print(time.time()-start)

    # Get AUC-ROC
    x_val_torch = torch.tensor(x_val, device=model.device, dtype=model.dtype).squeeze()
    y_val_torch = torch.tensor(y_val, device=model.device, dtype=torch.long).squeeze()

    model.eval()
    y_val_pred_proba_torch = model.predict_proba(x_val_torch)
    auc_val = roc_auc_score(y_val, y_val_pred_proba_torch[:, 1])
    print('val AUC-ROC: ' + str(auc_val))
    roc_list.append(auc_val)

    result_path_list.append(result_path)

res = pd.DataFrame()
res['weight_decay'] = weight_decay_list
res['roc_list'] = roc_list
res.to_csv('results_val_ROC_AUC__.csv', index=False)
print(res)
