import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from keras.preprocessing.sequence import pad_sequences

from utils import physionet_challenge_utility_script as pc

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

def plot_learning_curves(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('learning_curve.png')

def run_inception_model(i):
    # Inception Model
    inceptionModel = Inception()
    inception_model = inceptionModel.inception_model(input_shape=(5000, 12))
    # inception_model = inceptionModel.inception_model(input_shape=(5000, 12))
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
    history = inception_model.fit(
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

def get_X_train(X_train, order_array):
    X_train_final = []
    for i in order_array:
        data, header_data = pc.load_challenge_data(X_train[i])
        X_train_new = pad_sequences(data, maxlen=5000, truncating='post', padding="post")
        X_train_new = X_train_new.reshape(5000, 12)
        X_train_final.append(np.expand_dims(X_train_new, axis=0))

    return np.concatenate(X_train_final, axis=0)