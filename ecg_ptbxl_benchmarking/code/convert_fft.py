import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import wfdb

from utils import utils


def plot_ecg(ecg_data, title):
    
    data = ecg_data[0]
    sig_names = ecg_data[1]

    fig, axs = plt.subplots(4, 3, figsize=(16, 6))
    for lead in range(len(sig_names)):
        axs[lead % 4, lead // 4].plot(data[:, lead])
        axs[lead % 4, lead // 4].set_title(sig_names[lead])

    fig.suptitle(title)

    fig.savefig('ecg_sample.png')


def plot_ecg_fft(ecg_data, title, type='real'):

    if type=='real':
        data = ecg_data[0].real
    elif type=='imag':
        data = ecg_data[0].imag
    elif type=='abs':
        data = np.abs(ecg_data[0])

    freq = np.fft.fftfreq(data.shape[0])
    sig_names = ecg_data[1]

    fig, axs = plt.subplots(4, 3, figsize=(16, 6))
    for lead in range(len(sig_names)):
        axs[lead % 4, lead // 4].plot(freq, data[:, lead])
        axs[lead % 4, lead // 4].set_title(sig_names[lead])

    fig.suptitle(title)

    fig.savefig(f'ecg_sample_fft_{type}_freq.png')


datafolder = '../data/ptbxl/'
path = datafolder
# df = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

# data = utils.load_raw_data_ptbxl_fft(df, 500, path, 'abs')
# print(data.shape)

# sample = df.filename_hr.sample(random_state=37)
# data = wfdb.rdsamp(path+sample.values[0])
# data = data[0]

data = np.load(path+'raw500_imag.npy', allow_pickle=True)
sig_name = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# plot_ecg((data, sig_name), title=f'ECG Sample {sample.index[0]}')
sample = 91
plot_ecg((data[sample, :, :], sig_name), title=f'ECG Sample FFT Imag {sample}')

# data_fft = np.fft.fft(data, axis=0)

# plot_ecg_fft((data_fft, sig_name), title=f'ECG Sample {sample.index[0]}', type='real')
# plot_ecg_fft((data_fft, sig_name), title=f'ECG Sample {sample.index[0]}', type='imag')
# plot_ecg_fft((data_fft, sig_name), title=f'ECG Sample {sample.index[0]}', type='abs')

