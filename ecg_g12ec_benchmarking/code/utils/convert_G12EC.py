import os
import pandas as pd
import wfdb
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from scipy.io import loadmat
from stratisfy import stratisfy_df

output_folder = '../../data/G12ECRes'
output_datafolder_100 = output_folder + '/records100/'
output_datafolder_500 = output_folder + '/records500/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(output_datafolder_100):
    os.makedirs(output_datafolder_100)
if not os.path.exists(output_datafolder_500):
    os.makedirs(output_datafolder_500)


def store_as_wfdb(signame, data, sigfolder, fs):
    data = data.astype(np.float64)
    channel_itos = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    wfdb.wrsamp(signame,
                fs=fs,
                sig_name=channel_itos,
                p_signal=data,
                units=['mV'] * len(channel_itos),
                fmt=['16'] * len(channel_itos),
                write_dir=sigfolder)

df1 = pd.read_csv('../../SNOMED_mappings_scored.csv', sep=';')
df2 = pd.read_csv('../../SNOMED_mappings_unscored.csv', sep=';')
df_reference = pd.concat([df1, df2])
# label_dict = {1:'NORM', 2:'AFIB', 3:'1AVB', 4:'CLBBB', 5:'CRBBB', 6:'PAC', 7:'VPC', 8:'STD_', 9:'STE_'}

data = {'ecg_id': [], 'filename': [], 'validation': [], 'age': [], 'sex': [], 'scp_codes': []}

ecg_counter = 0
for folder in ['../../data']:
    filenames = os.listdir(folder)
    for filename in tqdm(filenames):
        if len(filename.split('.')) > 1 and filename.split('.')[1] == 'mat':
            ecg_counter += 1
            name = filename.split('.')[0]
            sig = loadmat(folder + '/' + filename)['val']
            with open(folder + '/' + name + '.hea', 'r') as f:
                metadata = f.readlines()
            data['ecg_id'].append(name)
            data['filename'].append(name)
            data['validation'].append(False)
            age = metadata[13].split(': ')[1].strip()
            data['age'].append(int(age) if age.lower() != 'nan' else 0)
            sex = metadata[14].split(': ')[1].strip()
            data['sex'].append(1 if sex == 'Male' else 0)
            # labels = df_reference[df_reference.Recording == name][['First_label' ,'Second_label' ,'Third_label']].values.flatten()
            scp_codes = metadata[15].split(': ')[1].strip().split(',')
            labels = []
            for code in scp_codes:
                labels.append(df_reference[df_reference['SNOMED CT Code'] == int(code)]['Abbreviation'].iloc[0])
            # data['scp_codes'].append({label_dict[key]:100 for key in labels})
            data['scp_codes'].append({key: 100 for key in labels})
            filename_to_be_saved = 'E' + str('0' * (5 - len(str(ecg_counter)))) + str(ecg_counter)
            store_as_wfdb(filename_to_be_saved, sig.T, output_datafolder_500, 500)
            down_sig = np.array([zoom(channel, .2) for channel in sig])
            store_as_wfdb(filename_to_be_saved, down_sig.T, output_datafolder_100, 100)

df = pd.DataFrame(data)
df['patient_id'] = df.ecg_id
df = stratisfy_df(df, 'strat_fold')
df.to_csv(output_folder + '/g12ec_database.csv')
