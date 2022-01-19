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

# pdb.set_trace()
sampling_frequency = 500
data_folder = '../../ecg_g12ec_benchmarking/data/'
#task = 'subdiagnostic'
task = 'diagnostic'

output_folder = '../output500/'
# For running subsets of data
# output_folder = '../output500_75PerData/'
# For running using fft
# output_folder = '../output500_100PerData_fft_real/'

train_folds = 9
print('Loading data started')

data, raw_labels = utils.load_dataset(data_folder, sampling_frequency) # compsig='real'
# pdb.set_trace()

# # padding
# data = np.array([pad_sequences(d.T, maxlen=5000, truncating='post', padding="post").T for d in data])
# data = np.concatenate([np.expand_dims(d, 0) for d in data], axis=0)

print(data.shape)
# pdb.set_trace()
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
# print(data.shape)
# pdb.set_trace()

labels.drop('strat_fold', axis=1, inplace=True)
labels = stratisfy.stratisfy_df(labels, 'strat_fold')

# group folds - added 12-11-21
# labels.rename(columns={'strat_fold': 'strat_fold_org'}, inplace=True)
# labels['strat_fold'] = labels['strat_fold_org'].apply(lambda x: int(np.ceil(x/2))) # 8-2 split
# labels['strat_fold'] = labels['strat_fold_org'].apply(lambda x: x % 3) # 7-3/6-4 split
#labels['strat_fold'] = labels['strat_fold_org'].apply(lambda x: x % 2) # 5-5 split

# n_folds = labels['strat_fold'].nunique()
# labels.to_csv(f'selectedData_{n_folds}folds.csv')

overall_result = []
label_aucs = []
label_supports = []
label_aucs_calibrated = []
overall_result_calibrated =[]
all_folds = sorted(list(labels.strat_fold.unique()))

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X_val, y_val):
        'Initialization'
        self.X = X_val
        self.y = y_val

    def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index,:]

# def plot_calibration_curve(name, fig_index, probs, class_i):
#     """Plot calibration curve for est w/o and with calibration. """
#
#     fig = plt.figure(fig_index, figsize=(10, 10))
#     ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 1), (2, 0))
#
#     ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#
#     frac_of_pos, mean_pred_value = calibration_curve(y_val[:, class_i], probs, n_bins=10)
#
#     ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f'{name}')
#     ax1.set_ylabel("Fraction of positives")
#     ax1.set_ylim([-0.05, 1.05])
#     ax1.legend(loc="lower right")
#     ax1.set_title(f'Calibration plot ({name})')
#
#     ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
#     ax2.set_xlabel("Mean predicted value")
#     ax2.set_ylabel("Count")
#     plt.savefig(calibration_folder + f"calibration_curve_class{class_i}.png")

def plot_calibration_curve(name, fig_index, probs, cal_probs, class_i):
    """Plot calibration curve for est w/o and with calibration. """

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    frac_of_pos, mean_pred_value = calibration_curve(y_val[:, class_i], probs, n_bins=10)
    frac_of_pos_cal, mean_pred_value_cal = calibration_curve(y_val[:, class_i], cal_probs, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f'{name}', linestyle='dashed', color='black')
    ax1.plot(mean_pred_value_cal, frac_of_pos_cal, "s-", label=f'{name} - Calibrated', color='black')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlabel("Class probability")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot for {name}')

    plt.savefig(calibration_folder + f"calibration_curve_class_{name.split(' ')[-1]}.png")

for test_fold in [all_folds[-1]]:
    train_folds = [fold for fold in all_folds if fold != test_fold]
    print(f'Test Fold: {test_fold}')

    # train_folds = [train_fold]
    # print(f'Train Fold: {train_folds}')

    X_train = data[labels.strat_fold.isin(train_folds)]
    y_train = Y[labels.strat_fold.isin(train_folds)]

    X_val = data[~(labels.strat_fold.isin(train_folds))]
    y_val = Y[~(labels.strat_fold.isin(train_folds))]
    

    num_classes = 18 # number of classes in G12EC
    # input_shape = [1000, 12]
    input_shape = [5000, 12]

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    # pdb.set_trace()

    experiment = 'exp1'
    modelname = 'fastai_xresnet1d101'
    pretrainedfolder = output_folder + experiment + '/models/' + modelname + '/'
    mpath = output_folder + f'finetune_g12ec_{experiment}' + '/'
    resultfolder = mpath + 'results/'
    # calibration_folder = mpath + 'calibration_curves/'
    calibration_folder = mpath + 'calibration_curves_tempscale/'
    n_classes_pretrained = 44 # <=== because we load the model from exp0, this should be fixed because this depends the experiment

    model = fastai_model(
        modelname, 
        num_classes, 
        sampling_frequency, 
        mpath, 
        input_shape=input_shape, 
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained, 
        # chunkify_valid = False,
        pretrained=True,
        epochs_finetuning=10,
        # lr=0.001,
        #bs=64,
    )

    standard_scaler = pickle.load(open(output_folder + experiment + '/data/standard_scaler.pkl', "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)
    # pdb.set_trace()

    model.fit(X_train, y_train, X_val, y_val)
    # model.__sklearn_is_fitted__
    # model.set_classes(mlb)

    y_val_pred = model.predict(X_val)
    # pdb.set_trace()

    # Plot calibration curves
    # for class_i in range(18):
    #     plot_calibration_curve(f"XResNet1d for Class {class_i}", 1, y_val_pred[:,class_i], class_i)

    # Perform calibration
    X_val = np.array([pad_sequences(d.T, maxlen=5000, truncating='post', padding="post").T for d in X_val])
    X_val = np.concatenate([np.expand_dims(d, 0) for d in X_val], axis=0)
    validation_set = Dataset(X_val, y_val)
    val_loader = DataLoader(validation_set, batch_size=700)
    scaled_model = ModelWithTemperature(model)
    scaled_model = scaled_model.set_temperature(val_loader)
    calibrated_probs = scaled_model.forward(X_val).detach().cpu().numpy()

    pdb.set_trace()
    # # print(mlb.inverse_transform(y_val), y_val.shape)
    # # pdb.set_trace()
    # calibrated_classifier = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    # calibrated_classifier.fit(X_val, y_val)
    # calibrated_probs = calibrated_classifier.predict_proba(X_val)
    # pdb.set_trace()
    # df_result_calibrated, aucs_calibrated = utils.evaluate_experiment(y_val, calibrated_probs)
    # label_aucs_calibrated.append(pd.DataFrame({'label': mlb.classes_, 'auc': aucs_calibrated}))
    # overall_result_calibrated.append(df_result_calibrated)
    for class_i in range(18):
        class_name = mlb.classes_[class_i]
        if "/" in class_name:
            class_name = class_name.replace("/", "_")
        plot_calibration_curve(class_name, 1, y_val_pred[:,class_i], calibrated_probs[:,class_i], class_i)

    try:
        df_result, aucs = utils.evaluate_experiment(y_val, y_val_pred)
    except Exception as e:
        print(e)
        df_result = pd.DataFrame({'macro_auc': 0}, index=[0])
        aucs = [0]*len(mlb.classes_)

    label_aucs.append(pd.DataFrame({'label': mlb.classes_, 'auc': aucs}))
    label_supports.append(pd.DataFrame(y_val, columns=mlb.classes_).sum(0).rename(f'support_{test_fold}'))
    # label_supports.append(pd.DataFrame(y_val, columns=mlb.classes_).sum(0).rename(f'support_{train_fold}'))

    overall_result.append(df_result)

# For saving calibrated results
# label_auc_df = [label_aucs_calibrated[i].rename(columns={'auc': 'auc_' + str(i + 1)}).set_index('label') for i in range(len(label_aucs_calibrated))]
# label_auc_df = label_auc_df[0].join([e for e in label_auc_df[1:]], how='inner')
# label_auc_df.to_csv(f'{resultfolder}{task}_{test_fold}thfold_label_aucs_calibrated.csv')
# pd.DataFrame({'macro_auc': [e['macro_auc'] for e in overall_result_calibrated]}, index=range(len(overall_result_calibrated))).to_csv(f'{resultfolder}{task}_{test_fold}thfold_macro_aucs_calibrated.csv')
# print(overall_result_calibrated)

'''
# For saving per-class AUC
label_auc_df = [label_aucs[i].rename(columns={'auc': 'auc_'+str(i+1)}).set_index('label') for i in range(len(label_aucs))]
label_auc_df = label_auc_df[0].join([e for e in label_auc_df[1:]], how='inner')
# label_auc_df.to_csv(f'{resultfolder}{task}_{test_fold}thfold_label_aucs.csv')
label_auc_df.to_csv(f'{resultfolder}{task}_inv_{train_fold}thfold_label_aucs.csv')

# For saving label supports
label_support_df = pd.concat(label_supports, axis=1)
# label_support_df.to_csv(f'{resultfolder}{task}_{test_fold}thfold_label_supports.csv')
label_support_df.to_csv(f'{resultfolder}{task}_inv_{train_fold}thfold_label_supports.csv')

# For saving macro AUC
# pd.DataFrame({'macro_auc': [e['macro_auc'] for e in overall_result]}, index=range(len(overall_result))).to_csv(f'{resultfolder}{task}_{test_fold}thfold_macro_aucs.csv')
pd.DataFrame({'macro_auc': [e['macro_auc'] for e in overall_result]}, index=range(len(overall_result))).to_csv(f'{resultfolder}{task}_inv_{train_fold}thfold_macro_aucs.csv')
# #
'''
print(overall_result)
