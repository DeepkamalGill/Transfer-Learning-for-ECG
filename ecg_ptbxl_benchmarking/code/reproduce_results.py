from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
import random


def main():
    
    datafolder = '../data/ptbxl/'
    #datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output500/'

    ############ FOR FFT EXPERIMENT ################
    # outputfolder = '../output500_100PerData_fft_real/'
    # fft_part = 'real'

    ############ FOR RUNNING SUBSETS OF DATA ################
    # percentage_of_data = 75
    # percentage_of_data = 25
    # percentage_of_data = 50
    percentage_of_data = 100
    # outputfolder = f'../output500_{percentage_of_data}PerData/'

    models = [
        conf_fastai_xresnet1d101,
        #conf_fastai_resnet1d_wang,
        #conf_fastai_lstm,
        #conf_fastai_lstm_bidir,
        #conf_fastai_fcn_wang,
        #conf_fastai_inception1d,
        #conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        #('exp0', 'all'),
        ('exp1', 'diagnostic'),
        # ('exp1.1', 'subdiagnostic'),
        # ('exp1.1.1', 'superdiagnostic'),
        #('exp2', 'form'),
        #('exp3', 'rhythm')
       ]

    random.seed(37)
    train_fold = random.sample(list(range(1, 11)),  int((percentage_of_data/100)*8))
    val_fold = random.sample([f for f in list(range(1, 11)) if f not in train_fold], 1)
    test_fold = [f for f in list(range(1, 11)) if f not in (train_fold+val_fold)]


    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, 
                            train_fold=train_fold, val_fold=val_fold, test_fold=test_fold, sampling_frequency=500)
        e.prepare()
        # e.prepare(percentage_of_data=None, compsig=fft_part)
        e.perform()
        e.evaluate(n_jobs = 2)

    # generate greate summary table
    utils.generate_ptbxl_summary_table()

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################
'''
    e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
    e.prepare()
    e.perform()
    e.evaluate()

    # generate greate summary table
    utils.ICBEBE_table()
'''
if __name__ == "__main__":
    main()
