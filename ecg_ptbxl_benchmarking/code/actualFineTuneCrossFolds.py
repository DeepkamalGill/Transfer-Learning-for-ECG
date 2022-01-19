import pickle
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.fastai_model import fastai_model
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from keras.preprocessing.sequence import pad_sequences
from utils.temperature_scaling import ModelWithTemperature
from torch.utils.data import ConcatDataset, DataLoader
import torch

from utils import utils
from utils import stratisfy

sampling_frequency = 500
data_folder = '../../ecg_g12ec_benchmarking/data/'
task = 'diagnostic'

PTBXL_SUBSET = False
PTBXL_FRAC = 75
INVERT_FOLDS = False
N_FOLDS = 10
RUN_ONCE = True

if not PTBXL_SUBSET:
    output_folder = '../output500/'
else:
    # For running on model trained on subsets of PTBXL
    output_folder = f'../output500_{PTBXL_FRAC}PerData/'

print('Loading data started')

data, raw_labels = utils.load_dataset(data_folder, sampling_frequency)
print(data.shape)
print("All dataset loaded")

# Map labels in PTB-XL to corresponding labels in G12EC
label_mapping = {'IAVB': '1AVB',
                 'AF': 'AFIB',
                 'AFL': 'AFLT',
                 #'CRBBB': 'CRBBB',
                 #'IRBBB': 'IRBBB',
                 'LAnFB': 'LAFB',
                 'LBBB': 'CLBBB',
                 'LQRSV': 'LVOLT',
                 'NSIVCB': 'IVCD',
                 'PR': 'PACE',
                 #'PAC': 'PAC',
                 #'LPR': 'LPR',
                 'LQT': 'LNGQT',
                 'QAb': 'QWAVE',
                 'SA': 'SARRH',
                 'SB': 'SBRAD',
                 'STach': 'STACH',
                 'SVPB': 'SVARR',
                 'TInv': 'INVT',
                 'IIAVB': '2AVB',
                 'AnMIs': 'ISCAN',
                 'AnMI': 'AMI',
                 'CHB': '3AVB', # low samples
                 #'ILBBB': 'ILBBB',
                 'IIs': 'ISCIN',
                 'LIs': 'ISCLA',
                 'LAE': 'LAO/LAE',
                 #'LPFB': 'LPFB',
                 #'LVH': 'LVH',
                 'MI': 'IMI', # low samples
                 'NSSTTA': 'NDT',
                 #'PSVT': 'PSVT',
                 'RAH': 'RAO/RAE',
                 #'RVH': 'RVH',
                 'STC': 'NST_', # low samples
                 'STD': 'STD_',
                 'STE': 'STE_',
                 'SVT': 'SVTAC',
                 'TIA': 'ISC_',
                 'VBig': 'BIGU',
                 'VH': 'SEHYP',
                 'VTrig': 'TRIGU'
                 #'WPW': 'WPW'
                }

def change_labels(ll):
    res_dict = ll.copy()
    for k,v in ll.items():
        if label_mapping.get(k) is not None:
            res_dict.update({label_mapping.get(k) : v})
            res_dict.pop(k, None)
    return res_dict
    
print("Data is loaded")
raw_labels.scp_codes = raw_labels.scp_codes.apply(change_labels)
raw_labels.scp_codes = raw_labels.scp_codes.apply(lambda x: {k: v for k, v in x.items() if k not in ['WPW', 'NST_', 'IMI', '3AVB']}) # removing classes with less samples

labels = utils.compute_label_aggregations(raw_labels, data_folder, task)

data, labels, Y, mlb = utils.select_data(data, labels, task, min_samples=0, outputfolder=output_folder)

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

# n_folds = labels['strat_fold'].nunique()
# labels.to_csv(f'selectedData_{n_folds}folds.csv')

overall_result = []
label_aucs = []
label_supports = []
all_folds = sorted(list(labels.strat_fold.unique()))

# for curr_fold in [all_folds[-1]]:
for curr_fold in all_folds:
    if RUN_ONCE:
        curr_fold = 10
    if not INVERT_FOLDS:
        train_folds = [fold for fold in all_folds if fold != curr_fold]
        test_folds = [curr_fold]
    else:
        train_folds = [curr_fold]
        test_folds = [fold for fold in all_folds if fold != curr_fold]
    
    print(f'Test Folds: {test_folds}')
    print(f'Train Folds: {train_folds}')

    X_train = data[labels.strat_fold.isin(train_folds)]
    y_train = Y[labels.strat_fold.isin(train_folds)]

    X_val = data[~(labels.strat_fold.isin(train_folds))]
    y_val = Y[~(labels.strat_fold.isin(train_folds))]
    
    num_classes = 18 # number of classes in G12EC
    input_shape = [5000, 12]

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    experiment = 'exp1'
    modelname = 'fastai_xresnet1d101'
    pretrainedfolder = output_folder + experiment + '/models/' + modelname + '/'
    mpath = output_folder + f'finetune_g12ec_{experiment}_crossfolds' + '/'
    resultfolder = mpath + 'results/'
    calibration_folder = mpath + 'calibration_curves/'
    n_classes_pretrained = 44

    model = fastai_model(
        modelname, 
        num_classes, 
        sampling_frequency, 
        mpath, 
        input_shape=input_shape, 
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained, 
        pretrained=True,
        epochs_finetuning=10,
    )

    standard_scaler = pickle.load(open(output_folder + experiment + '/data/standard_scaler.pkl', "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)

    model.fit(X_train, y_train, X_val, y_val)

    y_val_pred = model.predict(X_val)

    df_result, aucs = utils.evaluate_experiment(y_val, y_val_pred)

    label_aucs.append(pd.DataFrame({'label': mlb.classes_, f'auc_{curr_fold}': aucs}).set_index('label'))
    label_supports.append(pd.DataFrame(y_val, columns=mlb.classes_).sum(0).rename(f'support_{curr_fold}'))

    overall_result.append(df_result)

    if RUN_ONCE:

        # Plot calibration curves
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        colors = ['#A9A9A9', '#088F8F', '#0047AB', '#87CEEB', 'black']
        color_id = 0
        for class_i in range(18):
            class_name = mlb.classes_[class_i]
            if class_name in ['LNGQT', 'LVH', 'ISCLA', 'NDT', '1AVB']:
                frac_of_pos, mean_pred_value = calibration_curve(y_val[:, class_i], y_val_pred[:, class_i], n_bins=10)
                ax.plot(mean_pred_value, frac_of_pos, "s-", label=f'{class_name}', color=colors[color_id])
                ax.set_ylabel("Fraction of positives", fontsize=14)
                ax.set_xlabel("Class probability", fontsize=14)
                ax.set_ylim([-0.05, 1.05])
                ax.legend(loc="lower right", fontsize=13)
                ax.set_title(f'Calibration plot', fontsize=14)
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                color_id+=1

        plt.savefig(calibration_folder + "calibration_curve.png")
        break

# For saving per-class AUC
label_auc_df = label_aucs[0].join([e for e in label_aucs[1:]], how='inner')
label_auc_df.to_csv(f'{resultfolder}{task}{"_inv" if INVERT_FOLDS else ""}_{N_FOLDS}_folds_label_aucs.csv')

# For saving label supports
label_support_df = pd.concat(label_supports, axis=1)
label_support_df.to_csv(f'{resultfolder}{task}{"_inv" if INVERT_FOLDS else ""}_{N_FOLDS}_folds_label_supports.csv')

# For saving macro AUC
pd.DataFrame(
    {'macro_auc': [e['macro_auc'] for e in overall_result]}, 
    index=range(len(overall_result))
    ).to_csv(f'{resultfolder}{task}{"_inv" if INVERT_FOLDS else ""}_{N_FOLDS}_folds_macro_aucs.csv')

print(overall_result)