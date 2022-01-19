from utils import physionet_challenge_utility_script as pc
import numpy as np
import pdb
#import ecg_plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from keras.models import Sequential, Model
# from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from keras.layers.merge import concatenate
from scipy import optimize
from scipy.io import loadmat
import os
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV

from models.fastai_model import fastai_model
from utils.custom_utils import *
import models

icd_label_to_numerical_code = {
    'LAE'   : 67741000119109,
    'CRBBB' : 713427006,
    'ILBBB' : 251120003,
    'LQT'   : 111975006,
    #'STC'  : 55930002,
    'LAnFB' : 445118002,
    'IIAVB' : 195042002,
    'NSSTTA': 428750005,
    'IIs'   : 425419005,
    'VH'    : 266249003,
    'NSIVCB': 698252002,
    'RVH'   : 89792004,
    #'WPW'  : 74390002,
    'IRBBB' : 713426002,
    'AnMIs' : 426434006,
    'LPFB'  : 445211001,
    #'MI'   : 164865005,
    'IAVB'  : 270492004,
    'LVH'   : 164873001,
    'LIs'   : 425623009,
    #'CHB'  : 27885002,
    'LBBB'  : 164909002
}

data_file = '../selectedData.csv'

# Select the ECG IDs corresponding to 18 diagnostic classes
selected_ecg_ids = pd.read_csv(data_file).ecg_id.tolist()
custom_folds = pd.read_csv(data_file).strat_fold
invert_folds = False
run_suffix = f"{'inv_' if invert_folds else ''}{custom_folds.max()}folds"
result_folder = f'results/result_cross_folds_{run_suffix}/'

gender, age, labels, ecg_filenames = pc.import_key_data("./../data", icd_label_to_numerical_code, selected_ecg_ids)
ecg_filenames = np.asarray(ecg_filenames)

SNOMED_scored=pd.read_csv("./../SNOMED_mappings_scored.csv", sep=";")
SNOMED_unscored=pd.read_csv("./../SNOMED_mappings_unscored.csv", sep=";")
df_labels = pc.make_undefined_class(labels, pd.concat([SNOMED_scored, SNOMED_unscored]))
y, snomed_classes, mlb = pc.onehot_encode(df_labels)
y_all_comb = pc.get_labels_for_all_combinations(y)

print("Total number of unique combinations of diagnosis: {}".format(len(np.unique(y_all_comb))))

pdb.set_trace()
# INVERT_FOLDS = False
N_FOLDS = 10

if N_FOLDS == 10:
    labels.drop('strat_fold', axis=1, inplace=True)
    labels = stratisfy.stratisfy_df(labels, 'strat_fold')
else:
    # group folds - added 12-11-21
    labels.rename(columns={'strat_fold': 'strat_fold_org'}, inplace=True)
    if N_FOLDS == 5:
        labels['strat_fold'] = labels['strat_fold_org'].apply(lambda x: int(np.ceil(x/2))) # 8-2 split
    elif N_FOLDS == 3:
        labels['strat_fold'] = labels['strat_fold_org'].apply(lambda x: x % 3) # 7-3/6-4 split
    elif N_FOLDS == 2:
        labels['strat_fold'] = labels['strat_fold_org'].apply(lambda x: x % 2) # 5-5 split
    else:
        raise ValueError(f"The value of N_FOLDS can only be 2,3,5 and 10. Encountered value {N_FOLDS}")

overall_result = []
label_aucs = []
label_supports = []
folds = []
# all_folds = sorted(list(labels.strat_fold.unique()))

'''
if invert_folds:
    # just one fold in train
    for fold in range(custom_folds.nunique()):
        folds.append((np.array(custom_folds[custom_folds==(fold+1)].index), np.array(custom_folds[custom_folds!=(fold+1)].index)))
    print('Folds inverted !')
else:
    # just one fold in test
    for fold in range(custom_folds.nunique()):
        folds.append((np.array(custom_folds[custom_folds!=(fold+1)].index), np.array(custom_folds[custom_folds==(fold+1)].index)))

print(len(folds))
'''
# For getting class-wise AUCs
df = pd.DataFrame(columns = ['macro_auc'] + list(snomed_classes), index = ['Fold_'+str(i) for i in range(custom_folds.nunique())])
auc = []
label_aucs = []
label_supports = []
overall_result = []

# run_inception_model(i)

# XResNet config
conf_fastai_resnet1d101 = {'modelname': 'fastai_resnet1d101', 'modeltype': 'fastai_model',
                               'parameters': dict()}
modelname = conf_fastai_resnet1d101['modelname']
modelparams = conf_fastai_resnet1d101['parameters']
sampling_frequency = 500

for curr_fold in all_folds[-1]:
    if not INVERT_FOLDS:
        train_folds = [fold for fold in all_folds if fold != curr_fold]
        test_folds = [curr_fold]
    else:
        train_folds = [curr_fold]
        test_folds = [fold for fold in all_folds if fold != curr_fold]

    print(f'Test Folds: {test_folds}')
    print(f'Train Folds: {train_folds}')
    pdb.set_trace()

# for i in range(10): #range(10):
#     print("--------------------------------------------------")
#     print("All for fold {}".format(i))
#     print("--------------------------------------------------")
    order_array = folds[i][0]

    X_train = get_X_train(ecg_filenames, order_array)
    y_train = y[order_array]

    X_val, y_val = pc.generate_validation_data(ecg_filenames,y,folds[i][1])

    n_classes = y.shape[-1]
    input_shape = X_train[0].shape
    mpath = '../trained_models/' + f'model_{run_suffix}/' + modelname + '/'

    print(f'Train shape: {X_train.shape}, Test shape: {X_val.shape}')

    model = fastai_model(modelname, n_classes, sampling_frequency, mpath, input_shape, **modelparams, epochs=2)
    model.fit(X_train, y_train, X_val, y_val)

    x_val, y_val = pc.generate_validation_data(ecg_filenames,y,folds[i][1])

    y_pred = model.predict(X_val)

    macro_auc = evaluate_experiment(y_val, y_pred, average = 'macro')
    per_class_auc = evaluate_experiment(y_val, y_pred)
    print(macro_auc)
    # df.loc['Fold_'+str(i), 'macro_auc'] = macro_auc
    # df.loc['Fold_' + str(i), snomed_classes] = per_class_auc
    # df.to_csv(result_folder + 'macro_auc_score.csv')

    label_aucs.append(pd.DataFrame({'label': mlb.classes_, f'auc_{i}': per_class_auc}).set_index('label'))
    label_supports.append(pd.DataFrame(y_val, columns=mlb.classes_).sum(0).rename(f'support_{i}'))

    # overall_result.append(df_result)

# For saving per-class AUC
label_auc_df = label_aucs[0].join([e for e in label_aucs[1:]], how='inner')
label_auc_df.to_csv(f'{result_folder}diagnostic{"_inv" if invert_folds else ""}_{i}_folds_label_aucs.csv')

# For saving label supports
# label_support_df = pd.concat(label_supports, axis=1)
# label_support_df.to_csv(f'{resultfolder}_diagnostic_{"_inv" if invert_folds else ""}_{i}_folds_label_supports.csv')

# For saving macro AUC
# pd.DataFrame(
#     {'macro_auc': [e['macro_auc'] for e in overall_result]},
#     index=range(len(overall_result))
# ).to_csv(f'{resultfolder}_diagnostic_{"_inv" if invert_folds else ""}_{i}_folds_macro_aucs.csv')

# print(overall_result)
