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

selected_ecg_ids = pd.read_csv('../selectedData.csv').ecg_id.tolist()

gender, age, labels, ecg_filenames = pc.import_key_data("./../data/G12ECRes/records100", icd_label_to_numerical_code, selected_ecg_ids)
# print(data.shape)
# pdb.set_trace()
ecg_filenames = np.asarray(ecg_filenames)
#pdb.set_trace()

SNOMED_scored=pd.read_csv("./../SNOMED_mappings_scored.csv", sep=";")
SNOMED_unscored=pd.read_csv("./../SNOMED_mappings_unscored.csv", sep=";")
df_labels = pc.make_undefined_class(labels, pd.concat([SNOMED_scored, SNOMED_unscored]))
#pdb.set_trace()
y , snomed_classes = pc.onehot_encode(df_labels)
#pdb.set_trace()
y_all_comb = pc.get_labels_for_all_combinations(y)
#pdb.set_trace()

print("Total number of unique combinations of diagnosis: {}".format(len(np.unique(y_all_comb))))

folds = pc.split_data(labels, y_all_comb)
print(type(folds[0][1]))
class Inception:
    def __init__(self):
        pass

    def inception_block(self, prev_layer):
        conv1 = Conv1D(filters=64, kernel_size=1, padding='same')(prev_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        conv3 = Conv1D(filters=64, kernel_size=1, padding='same')(prev_layer)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv1D(filters=64, kernel_size=3, padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        conv5 = Conv1D(filters=64, kernel_size=1, padding='same')(prev_layer)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv1D(filters=64, kernel_size=5, padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)

        pool = MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
        convmax = Conv1D(filters=64, kernel_size=1, padding='same')(pool)
        convmax = BatchNormalization()(convmax)
        convmax = Activation('relu')(convmax)

        layer_out = concatenate([conv1, conv3, conv5, convmax], axis=1)

        return layer_out

    def inception_model(self, input_shape):
        X_input = Input(input_shape)

        X = ZeroPadding1D(3)(X_input)

        X = Conv1D(filters=64, kernel_size=7, padding='same')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPool1D(pool_size=3, strides=2, padding='same')(X)

        X = Conv1D(filters=64, kernel_size=1, padding='same')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = self.inception_block(X)
        X = self.inception_block(X)

        X = MaxPool1D(pool_size=7, strides=2, padding='same')(X)

        X = GlobalAveragePooling1D()(X)
        X = Dense(18, activation='sigmoid')(X)

        model = Model(inputs=X_input, outputs=X, name='Inception')

        return model

df = pd.DataFrame(columns = ['macro_auc'] + list(snomed_classes), index = ['Fold_'+str(i) for i in range(10)])

'''
for i in range(10):
    print("--------------------------------------------------")
    print("All for fold {}".format(i))
    print("--------------------------------------------------")
    order_array = folds[i][0]

    def shuffle_batch_generator(batch_size, gen_x, gen_y):
        np.random.shuffle(order_array)
        batch_features = np.zeros((batch_size, 5000, 12))
        batch_labels = np.zeros((batch_size, snomed_classes.shape[0]))  # drop undef class
        while True:
            for i in range(batch_size):
                batch_features[i] = next(gen_x)
                batch_labels[i] = next(gen_y)

            yield batch_features, batch_labels

    def generate_y_shuffle(y_train):
        while True:
            for i in order_array:
                y_shuffled = y_train[i]
                yield y_shuffled


    def generate_X_shuffle(X_train):
        while True:
            for i in order_array:
                # if filepath.endswith(".mat"):
                data, header_data = pc.load_challenge_data(X_train[i])
                X_train_new = pad_sequences(data, maxlen=5000, truncating='post', padding="post")
                X_train_new = X_train_new.reshape(5000, 12)
                yield X_train_new


    def thr_chall_metrics(thr, label, output_prob):
        return -pc.compute_challenge_metric_for_opt(label, np.array(output_prob > thr))


    def load_challenge_data(filename):
        x = loadmat(filename)
        data = np.asarray(x['val'], dtype=np.float64)
        new_file = filename.replace('.mat', '.hea')
        input_header_file = os.path.join(new_file)
        with open(input_header_file, 'r') as f:
            header_data = f.readlines()
        return data, header_data


    def generate_validation_data(ecg_filenames, y, test_order_array):
        y_train_gridsearch = y[test_order_array]
        ecg_filenames_train_gridsearch = ecg_filenames[test_order_array]

        ecg_train_timeseries = []
        for names in ecg_filenames_train_gridsearch:
            data, header_data = load_challenge_data(names)
            data = pad_sequences(data, maxlen=5000, truncating='post', padding="post")
            ecg_train_timeseries.append(data)
        X_train_gridsearch = np.asarray(ecg_train_timeseries)

        X_train_gridsearch = X_train_gridsearch.reshape(ecg_filenames_train_gridsearch.shape[0], 5000, 12)

        return X_train_gridsearch, y_train_gridsearch


    def compute_modified_confusion_matrix(labels, outputs):
        # Compute a binary multi-class, multi-label confusion matrix, where the rows
        # are the labels and the columns are the outputs.
        num_recordings, num_classes = np.shape(labels)
        A = np.zeros((num_classes, num_classes))

        # Iterate over all of the recordings.
        for i in range(num_recordings):
            # Calculate the number of positive labels and/or outputs.
            normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
            # Iterate over all of the classes.
            for j in range(num_classes):
                # Assign full and/or partial credit for each positive class.
                if labels[i, j]:
                    for k in range(num_classes):
                        if outputs[i, k]:
                            A[j, k] += 1.0 / normalization

        return A


    def plot_normalized_conf_matrix_dev(y_pred, ecg_filenames, y, val_fold, threshold, snomedclasses):
        df_cm = pd.DataFrame(compute_modified_confusion_matrix(generate_validation_data(ecg_filenames, y, val_fold)[1],
                                                               (y_pred > threshold) * 1), columns=snomedclasses,
                             index=snomedclasses)
        df_cm = df_cm.fillna(0)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        df_norm_col = (df_cm - df_cm.mean()) / df_cm.std()
        plt.figure(figsize=(36, 14))
        sns.set(font_scale=1.4)
        sns.heatmap(df_norm_col, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt=".2f", cbar=False)  # font size


    def apply_thresholds(preds, thresholds):
        """
            apply class-wise thresholds to prediction score in order to get binary format.
            BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
        """
        tmp = []
        for p in preds:
            tmp_p = (p > thresholds).astype(int)
            if np.sum(tmp_p) == 0:
                tmp_p[np.argmax(p)] = 1
            tmp.append(tmp_p)
        tmp = np.array(tmp)
        return tmp


    def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=False):
        f_beta = 0
        g_beta = 0
        if single:  # if evaluating single class in case of threshold-optimization
            sample_weights = np.ones(y_true.sum(axis=1).shape)
        else:
            sample_weights = y_true.sum(axis=1)
        for classi in range(y_true.shape[1]):
            y_truei, y_predi = y_true[:, classi], y_pred[:, classi]
            TP, FP, TN, FN = 0., 0., 0., 0.
            for i in range(len(y_predi)):
                sample_weight = sample_weights[i]
                if y_truei[i] == y_predi[i] == 1:
                    TP += 1. / sample_weight
                if ((y_predi[i] == 1) and (y_truei[i] != y_predi[i])):
                    FP += 1. / sample_weight
                if y_truei[i] == y_predi[i] == 0:
                    TN += 1. / sample_weight
                if ((y_predi[i] == 0) and (y_truei[i] != y_predi[i])):
                    FN += 1. / sample_weight
            f_beta_i = ((1 + beta1 ** 2) * TP) / ((1 + beta1 ** 2) * TP + FP + (beta1 ** 2) * FN)
            g_beta_i = (TP) / (TP + FP + beta2 * FN)

            f_beta += f_beta_i
            g_beta += g_beta_i

        return {'F_beta_macro': f_beta / y_true.shape[1], 'G_beta_macro': g_beta / y_true.shape[1]}


    def evaluate_experiment(y_true, y_pred, thresholds=None, average = None):
        results = {}
        if not thresholds is None:
            # binary predictions
            y_pred_binary = apply_thresholds(y_pred, thresholds)
            # PhysioNet/CinC Challenges metrics
            challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
            results['F_beta_macro'] = challenge_scores['F_beta_macro']
            results['G_beta_macro'] = challenge_scores['G_beta_macro']
        # label based metric
        macro_auc_or_per_class = roc_auc_score(y_true, y_pred, average= average)
        return macro_auc_or_per_class


    inceptionModel = Inception()
    inception_model = inceptionModel.inception_model(input_shape=(5000, 12))
    inception_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            metrics=[tf.keras.metrics.BinaryAccuracy(
                                name='accuracy', dtype=None, threshold=0.5), tf.keras.metrics.Recall(name='Recall'),
                                tf.keras.metrics.Precision(name='Precision'),
                                tf.keras.metrics.AUC(
                                    num_thresholds=200,
                                    curve="ROC",
                                    summation_method="interpolation",
                                    name="AUC",
                                    dtype=None,
                                    thresholds=None,
                                    multi_label=True,
                                    label_weights=None,
                                )])

    batchsize = 10
    inception_model.fit(
        x=shuffle_batch_generator(
            batch_size=batchsize,
            gen_x=generate_X_shuffle(ecg_filenames),
            gen_y=generate_y_shuffle(y)),
        epochs=50,
        steps_per_epoch=(len(order_array) / (batchsize * 10)),
        validation_data=pc.generate_validation_data(
            ecg_filenames,
            y,
            folds[i][1])
    )
    x_val, y_val = pc.generate_validation_data(ecg_filenames,y,folds[i][1])
    y_pred = inception_model.predict(x_val)
    macro_auc = evaluate_experiment(y_val, y_pred, average = 'macro')
    per_class_auc = evaluate_experiment(y_val, y_pred)
    df.loc['Fold_'+str(i), 'macro_auc'] = macro_auc
    df.loc['Fold_' + str(i), snomed_classes ] = per_class_auc
    df.to_csv('Macro_auc_score_for_10_folds.csv')
    print(df.loc['Fold_' + str(i)])
'''
selected_data = pd.read_csv('../selectedData.csv')
test_fold_ids = np.array([i for i in selected_data.index.to_list() if selected_data.strat_fold.iloc[i]==10])
train_fold_ids = np.array([i for i in selected_data.index.to_list() if selected_data.strat_fold.iloc[i]!=10])
# test_data = data[test_fold_ids,:,:]
# train_data = data[test_fold_ids,:,:]

def shuffle_batch_generator(batch_size, gen_x, gen_y):
    np.random.shuffle(train_fold_ids)
    batch_features = np.zeros((batch_size, 1000, 12))
    batch_labels = np.zeros((batch_size, snomed_classes.shape[0]))  # drop undef class
    while True:
        for i in range(batch_size):
            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)

        yield batch_features, batch_labels


def generate_y_shuffle(y_train):
    while True:
        for i in train_fold_ids:
            y_shuffled = y_train[i]
            yield y_shuffled


def generate_X_shuffle(X_train):
    while True:
        for i in train_fold_ids:
            # if filepath.endswith(".mat"):
            data, header_data = pc.load_challenge_data(X_train[i])
            X_train_new = pad_sequences(data, maxlen=1000, truncating='post', padding="post")
            X_train_new = X_train_new.reshape(1000, 12)
            yield X_train_new

def evaluate_experiment(y_true, y_pred, thresholds=None, average = None):
    results = {}
    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']
    # label based metric
    macro_auc_or_per_class = roc_auc_score(y_true, y_pred, average= average)
    return macro_auc_or_per_class

inceptionModel = Inception()
inception_model = inceptionModel.inception_model(input_shape=(1000, 12))
inception_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        metrics=[tf.keras.metrics.BinaryAccuracy(
                            name='accuracy', dtype=None, threshold=0.5), tf.keras.metrics.Recall(name='Recall'),
                            tf.keras.metrics.Precision(name='Precision'),
                            tf.keras.metrics.AUC(
                                num_thresholds=200,
                                curve="ROC",
                                summation_method="interpolation",
                                name="AUC",
                                dtype=None,
                                thresholds=None,
                                multi_label=True,
                                label_weights=None,
                            )])

batchsize = 10
inception_model.fit(
    x=shuffle_batch_generator(
        batch_size=batchsize,
        gen_x=generate_X_shuffle(ecg_filenames),
        gen_y=generate_y_shuffle(y)),
    epochs=50,
    steps_per_epoch=(len(train_fold_ids) / (batchsize * 10)),
    validation_data=pc.generate_validation_data(
        ecg_filenames,
        y,
        test_fold_ids)
)
x_val, y_val = pc.generate_validation_data(ecg_filenames,y,test_fold_ids)
y_pred = inception_model.predict(x_val)
macro_auc = evaluate_experiment(y_val, y_pred, average = 'macro')
per_class_auc = evaluate_experiment(y_val, y_pred)
print(macro_auc, per_class_auc)
df_new = pd.DataFrame({'labels': snomed_classes, 'auc': per_class_auc})
df.to_csv('Macro_auc_score_for_10_folds.csv')
df_new.to_csv('per_class_auc_G12EC_independent.csv', index=False)
