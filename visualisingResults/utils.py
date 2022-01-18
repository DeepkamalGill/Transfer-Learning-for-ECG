
import os
import glob as g
import sys
from collections import Counter
from ast import literal_eval
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    now = os.getcwd()
    FONT_SIZE = 12
    #------------------------------------------------
    # Plotting Graphs of g12EC independent run - Subsets
    # ------------------------------------------------
    
    folder_name = '/w/247/deepkamal/ptb_xl/ecg_g12ec_benchmarking/code/results'
    os.chdir(folder_name)
    lofs = g.glob('*')
    print(lofs)

    macro_aucs_g12ec_independent = {}
    for folder_now in lofs:
        os.chdir(folder_now)
        try:
            df = pd.read_csv('macro_auc_score.csv')
            mac_auc = df.iloc[0, 1] # 'Fold_0', 'macro_auc'
            macro_aucs_g12ec_independent.update({folder_now.split('/')[-1]: mac_auc})
        except Exception as e:
            pass
        os.chdir(folder_name)
    os.chdir(now)
    plt.bar(macro_aucs_g12ec_independent.keys(), macro_aucs_g12ec_independent.values())
    print('Completed plotting G12EC independent')
    # ------------------------------------------------
    # Plotting Graphs of g12EC Finetune run - Subsets
    # ------------------------------------------------
    folder_name = '/w/247/deepkamal/ptb_xl/ecg_ptbxl_benchmarking/output500/finetune_g12ec_exp1_crossfolds/results'
    os.chdir(folder_name)

    print('Starting plotting G12EC finetune')
    lofs = g.glob('*_macro_aucs.csv')
    print(lofs)

    macro_aucs_g12ec_finetuned = {}
    for file_now in lofs:
        try:
            df = pd.read_csv(file_now)
            mac_auc = df['macro_auc'].apply(lambda x: float(x.split(' '*4)[1].split('\n')[0])).mean()
            # mac_auc = float(df.iloc[0, 1].split('    ')[1].split('\n')[0])
            macro_aucs_g12ec_finetuned.update({file_now.split('/')[-1]: mac_auc})
        except Exception as e:
            pass
    plt.bar(macro_aucs_g12ec_finetuned.keys(), macro_aucs_g12ec_finetuned.values())

    # ------------------------------------------------
    # Saving the  overall graph
    # ------------------------------------------------
    os.chdir(now)
    df1 = pd.DataFrame(macro_aucs_g12ec_independent, index=[0])
    df1 = df1.T
    dd1 = {'result_inv_5folds': 20, 'result_inv_10folds': 10, 'result_5folds': 80, 'result_inv_2folds': 30,
           'result_10folds': 90, 'result_2folds': 70, 'result_1folds':50}
    df1.loc[:, 'fraction_of_training_data'] = [dd1.get(i) for i in df1.index.tolist()]
    df2 = pd.DataFrame(macro_aucs_g12ec_finetuned, index=[0])
    df2 = df2.T
    # dd2 = {'diagnostic_inv_10thfold_macro_aucs.csv': 10, 'diagnostic_5thfold_macro_aucs.csv': 80,
    #        'diagnostic_2thfold_macro_aucs.csv': 70, 'diagnostic_inv_2thfold_macro_aucs.csv': 30,
    #        'diagnostic_1thfold_macro_aucs.csv': 50, 'diagnostic_10thfold_macro_aucs.csv': 90,
    #        'diagnostic_inv_5thfold_macro_aucs.csv': 20}
    dd2 = {'diagnostic_inv_10_folds_macro_aucs.csv': 10, 'diagnostic_5_folds_macro_aucs.csv': 80,
           'diagnostic_3_folds_macro_aucs.csv': 70, 'diagnostic_inv_3_folds_macro_aucs.csv': 30,
           'diagnostic_2_folds_macro_aucs.csv': 50, 'diagnostic_10_folds_macro_aucs.csv': 90,
           'diagnostic_inv_5_folds_macro_aucs.csv': 20}
    df2.loc[:, 'fraction_of_training_data'] = [dd2.get(i) for i in df2.index.tolist()]
    df1.columns = ['g12ec_independent', 'fraction_of_training_data']
    df2.columns = ['g12ec_finetuned', 'fraction_of_training_data']
    merged_res = pd.merge(df1, df2, on = 'fraction_of_training_data')
    merged_res = merged_res.sort_values('fraction_of_training_data')
    merged_res.fraction_of_training_data = merged_res.fraction_of_training_data.astype(str)
    merged_res.plot.bar(x="fraction_of_training_data", y=['g12ec_independent' , 'g12ec_finetuned'], color = ['#87CEEB', '#4682B4'], edgecolor = 'white')

    plt.ylabel('Macro ROC - AUC', fontsize=FONT_SIZE)
    plt.xlabel('% of Training Data', fontsize=FONT_SIZE)
    plt.legend(['G12EC independent', 'G12EC fine-tuned'], loc='upper right', fontsize=10)
    plt.ylim(0.5, 1)
    plt.xticks(rotation=1, fontsize=FONT_SIZE)
    plt.yticks(np.arange(0.5, 1.05, 0.05), fontsize=FONT_SIZE)
    plt.title('Model Results using G12EC subsets', fontsize=FONT_SIZE)
    plt.savefig('results_of_g12ec_independent_and_finetune.png')

    # ------------------------------------------------
    # Plotting Graphs of ptbxl independent run - Subsets
    # ------------------------------------------------
    folder_of_concern = ['output500', 'output500_25PerData', 'output500_50PerData', 'output500_75PerData']
    folder_path = '/w/247/deepkamal/ptb_xl/ecg_ptbxl_benchmarking/'
    path_inside_folder_actual_dataset = '/exp1/models/fastai_xresnet1d101/results/'
    results_filename = 'te_results_new.csv'
    path_inside_folder_finetuned = '/finetune_g12ec_exp1_crossfolds/results/'
    filename_for_macro_auc_finetuned = 'diagnostic_inv_10_folds_macro_aucs.csv'

    macro_aucs_ptbxl_actual_data = {}
    macro_aucs_g12ec_finetuned_ptbxl_reduced_data = {}
    macro_aucs_g12ec_finetuned_ptbxl_reduced_data_per_label = {}
    for folder_now in folder_of_concern:
        overall_path_for_actual_data_training = folder_path + folder_now + path_inside_folder_actual_dataset
        overall_path_for_finetuned_data = folder_path + folder_now + path_inside_folder_finetuned
        try:
            df = pd.read_csv(overall_path_for_actual_data_training + results_filename)
            mac_auc = df.auc.mean()
            macro_aucs_ptbxl_actual_data.update({folder_now: mac_auc})
        except Exception as e:
            pass
        try:
            df = pd.read_csv(overall_path_for_finetuned_data + filename_for_macro_auc_finetuned)
            mac_auc = df['macro_auc'].apply(lambda x: float(x.split(' '*4)[1].split('\n')[0])).mean()
            macro_aucs_g12ec_finetuned_ptbxl_reduced_data.update({folder_now: mac_auc})
        except Exception as e:
            pass
        try:
            df = pd.read_csv(overall_path_for_finetuned_data + filename_for_micro_auc_finetuned)
            macro_aucs_g12ec_finetuned_ptbxl_reduced_data_per_label.update({folder_now: df})
        except Exception as e:
            pass
    os.chdir(now)
    print('Completed collecting ptbxl independent and finetuned with reduced data')

    # ------------------------------------------------
    # Plotting Graphs of ptbxl independent run - Subsets
    # ------------------------------------------------
    df1 = pd.DataFrame(macro_aucs_g12ec_finetuned_ptbxl_reduced_data, index = [0])
    df1 = df1.T
    df1.columns = ['g12ec_finetuned_on_ptbxl_reduced']

    df2 = pd.DataFrame(macro_aucs_ptbxl_actual_data, index=[0])
    df2 = df2.T
    df2.columns = ['ptbxl_mac_aucs']

    merged_ptbxl_and_g12ec_reduced_dataset = pd.concat([df1, df2], axis = 1)
    merged_ptbxl_and_g12ec_reduced_dataset.index = [100, 25, 50, 75]
    merged_ptbxl_and_g12ec_reduced_dataset = merged_ptbxl_and_g12ec_reduced_dataset.sort_index()
    merged_ptbxl_and_g12ec_reduced_dataset.plot.bar(color = ['#4682B4', '#088F8F'], edgecolor = 'white')

    plt.hlines(0.655, xmin=-20, xmax=100, linestyles='dotted', colors='black', label='G12EC independent (10%)')
    plt.ylim([0.5, 1])
    plt.xticks(rotation=1, fontsize=FONT_SIZE)
    plt.yticks(np.arange(0.5, 1.05, 0.05), fontsize=FONT_SIZE)
    plt.ylabel('Macro ROC - AUC', fontsize=FONT_SIZE)
    plt.xlabel('% of PTB-XL used for training', fontsize=FONT_SIZE)
    plt.legend(['G12EC independent (10%)', 'G12EC finetuned','PTB-XL'], loc='lower left', fontsize=10)
    plt.title('Model Results with PTB-XL subsets', fontsize=FONT_SIZE)
    plt.savefig('ptbxl_and_g12ec_on_reduced_dataset.png')

    # # ------------------------------------------------
    # # Plotting Graphs of ptbxl independent run - Subsets
    # # ------------------------------------------------
    # df100 = macro_aucs_g12ec_finetuned_ptbxl_reduced_data_per_label.get('output500')
    # df100.columns = ['labels', 100]
    # df25 = macro_aucs_g12ec_finetuned_ptbxl_reduced_data_per_label.get('output500_25PerData')
    # df25.columns = ['labels', 25]
    # df50 = macro_aucs_g12ec_finetuned_ptbxl_reduced_data_per_label.get('output500_50PerData')
    # df50.columns = ['labels', 50]
    # df75 = macro_aucs_g12ec_finetuned_ptbxl_reduced_data_per_label.get('output500_75PerData')
    # df75.columns = ['labels', 75]

    # dfs = [df100, df25, df50, df75]
    # merged_micro_aucs = pd.merge(df100, df25, on='labels')
    # merged_micro_aucs = pd.merge(merged_micro_aucs, df50, on='labels')
    # merged_micro_aucs = pd.merge(merged_micro_aucs, df75, on='labels')

    # merged_micro_aucs = merged_micro_aucs.T
    # merged_micro_aucs.columns = merged_micro_aucs.loc['labels', :]
    # merged_micro_aucs.drop(['labels'], axis = 0, inplace = True)
    # #merged_micro_aucs = merged_micro_aucs.loc[:, ['LVH', 'LAFB', 'IRBBB', '1AVB', 'IVCD']]
    # merged_micro_aucs = merged_micro_aucs.sort_index()
    # merged_micro_aucs.T.plot.bar()
    # plt.ylim([0.6, 1])
    # plt.savefig('Per_class_AUC_finetuned_data.png')

    # ------------------------------------------------
    # Per class AUC for finetuned dataset
    # ------------------------------------------------
    FONT_SIZE = 15
    path_to_g12ec_dataset_selected = '/w/247/deepkamal/ptb_xl/ecg_g12ec_benchmarking/selectedData.csv'
    path_to_test_data_micro_aucs_g12ec_independent = '/w/247/deepkamal/ptb_xl/ecg_g12ec_benchmarking/code/results/result_10folds/macro_auc_score.csv'
    path_to_test_data_micro_auc_g12ec_finetuned = '/w/247/deepkamal/ptb_xl/ecg_ptbxl_benchmarking/output500/finetune_g12ec_exp1_crossfolds/results/diagnostic_10_folds_label_aucs.csv'
    df1 = pd.read_csv(path_to_g12ec_dataset_selected)
    df2 = pd.read_csv(path_to_test_data_micro_aucs_g12ec_independent)
    df3 = pd.read_csv(path_to_test_data_micro_auc_g12ec_finetuned)

    ll = [j for i in df1.diagnostic.tolist() for j in literal_eval(i)]
    counter = dict(Counter(ll))
    per_class_auc_g12_independednt = df2.iloc[0, 2:]
    df3.index = df3.label
    per_class_auc_g12_finetuned = df3.loc[:, 'auc_10']
    label_ptbxl_to_numeric_code = {
        '1AVB' : 270492004,
        '2AVB' : 195042002,
        'CLBBB' : 164909002,
        'CRBBB' : 713427006,
        'ILBBB' : 251120003,
        'IRBBB' : 713426002,
        'AnMIs' : 426434006,
        'IIs' : 425419005,
        'ISCLA' : 425623009,
        'IVCD' : 698252002,
        'LAFB' : 445118002,
        'LAO/LAE' : 67741000119109,
        'LNGQT' : 111975006,
        'LPFB' : 445211001,
        'LVH' : 164873001,
        'NDT' : 428750005,
        'RVH' : 89792004,
        'SEHYP' : 266249003
    }
    rev_map = {}
    for k,v in label_ptbxl_to_numeric_code.items():
        rev_map.update({str(v):k})
    per_class_auc_g12_independednt.index = [rev_map.get(i) for i in per_class_auc_g12_independednt.index.tolist()]
    counter.update({'IIs': counter.get('ISCIN')})
    counter.pop('ISCIN')
    counter.update({'AnMIs': counter.get('ISCAN')})
    counter.pop('ISCAN')
    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
    val = counter.values()
    sum_val = sum(val)
    val = [i/sum_val for i in val]
    plt.clf()

    fig, ax1 = plt.subplots(figsize=(20, 9))

    ax1.bar(counter.keys(), height = val, alpha=0.6)
    ax1.set_ylabel('% Count per class', fontsize=FONT_SIZE)

    ax2 = ax1.twinx()
    per_class_auc_g12_independednt = per_class_auc_g12_independednt[counter.keys()]
    ax2.plot(per_class_auc_g12_independednt.index, per_class_auc_g12_independednt.values, linestyle='dashed', color='#0047AB')
    # per_class_auc_g12_independednt.plot.line(secondary_y = True , style = '--', color = '#0047AB')

    new_naming = []
    for ind_names in per_class_auc_g12_finetuned.index:
        if ind_names == 'ISCAN':
            new_naming.append('AnMIs')
        elif ind_names == 'ISCIN':
            new_naming.append('IIs')
        else:
            new_naming.append(ind_names)

    per_class_auc_g12_finetuned.index = new_naming
    per_class_auc_g12_finetuned.index.name = 'Diagnostic class'
    per_class_auc_g12_finetuned = per_class_auc_g12_finetuned[counter.keys()]
    ax2.plot(per_class_auc_g12_finetuned, color = '#0047AB')
    # ax2 = per_class_auc_g12_finetuned.plot.line(secondary_y = True)
    ax2.set_ylabel('ROC - AUC', fontsize=FONT_SIZE)

    labels = list(per_class_auc_g12_independednt.index)
    plt.xticks(rotation=90, labels = labels, ticks = range(len(labels)), fontsize=FONT_SIZE)

    plt.legend([ 'G12EC finetuned', 'G12EC independent'], loc='lower left', fontsize=13)
    plt.ylim([0.5, 1])
    ax1.set_xlabel('Diagnostic Classes', fontsize=FONT_SIZE)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE)
    ax1.tick_params(axis='y', labelsize=FONT_SIZE)
    ax2.tick_params(axis='y', labelsize=FONT_SIZE)
    # ax1.set_yticklabels(fontsize=FONT_SIZE)
    # ax2.set_yticks()
    # ax2.set_yticklabels(np.arange(0.5, 1.05, 0.05), fontsize=FONT_SIZE)
    plt.title('Label AUCs on G12EC dataset', fontsize=FONT_SIZE)

    plt.savefig('label_distribution_auc_per_class_ptbxl_finetuned.png')

    # ------------------------------------------------
    # Data Distribution
    # ------------------------------------------------
    '''
    icd_label_to_numerical_code = {
        'LAE': 67741000119109,
        'CRBBB': 713427006,
        'ILBBB': 251120003,
        'LQT': 111975006,
        # 'STC'  : 55930002,
        'LAnFB': 445118002,
        'IIAVB': 195042002,
        'NSSTTA': 428750005,
        'IIs': 425419005,
        'VH': 266249003,
        'NSIVCB': 698252002,
        'RVH': 89792004,
        # 'WPW'  : 74390002,
        'IRBBB': 713426002,
        'AnMIs': 426434006,
        'LPFB': 445211001,
        # 'MI'   : 164865005,
        'IAVB': 270492004,
        'LVH': 164873001,
        'LIs': 425623009,
        # 'CHB'  : 27885002,
        'LBBB': 164909002
    }
    filename = 'SNOMED_all.csv'
    df = pd.read_csv(filename)
    ll = list(icd_label_to_numerical_code.keys())
    def check(x):
        return True if x in ll else False
    df = df[df.Abbreviation.apply(check)]
    dd = df.loc[:, ['Abbreviation', 'PTB-XL', 'Georgia']]
    dd.index = dd.Dx
    dd = dd.iloc[:, 1:]
    sns.heatmap(dd, annot=True, fmt="", cmap="Blues")
    dd1 = dd.div(dd.sum(axis=0), axis=1).round(3)
    sns.heatmap(dd1, annot=True, fmt="", cmap="YlGnBu")
    plt.savefig('DataDistribution.png')
    '''

    # comparison across 100/500 Hz - Exp 5
    df = pd.DataFrame({'100Hz': [0.9278, 0.49, 0.48], '500Hz': [0.947, 0.9186, 0.843]}, index=['PTB-XL', 'G12EC fine-tuned', 'G12EC independent'])
    
    pos = np.arange(df.shape[0])
    width=0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(pos - 0.2, height=df['100Hz'], width=width, color=['#088F8F', '#4682B4', '#87CEEB'], label='100Hz', edgecolor='white', hatch='///')
    rects2 = ax.bar(pos + 0.2, height=df['500Hz'], width=width, color=['#088F8F', '#4682B4', '#87CEEB'], label='500Hz')
    ax.set_ylabel('Macro ROC - AUC', fontsize=FONT_SIZE)

    ax.legend(loc='lower left', fontsize=10)
    ax.set_title('Comparison across frequencies', fontsize=FONT_SIZE)
    ax.set_xticks(pos)
    ax.set_xticklabels(df.index, fontsize=FONT_SIZE)
    fig.savefig('performance_across_frequencies.png')