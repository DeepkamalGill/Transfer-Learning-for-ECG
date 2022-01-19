# Transfer Learning for ECG Classification using PTB-XL

Electrocardiography is a clinical procedure that records the activity of the heart at rest. ECG signals are vital for gaining information about the heart rate, rhythm as well as diagnosis of conditions such as hypertension and myocardial infarction. Automated interpretation of these signals can help in early detection of heart diseases, which can in turn, prevent severe outcomes such as arrhythmia. However, its applicability is limited due to the lack of large-scale datasets. Transfer learning presents a viable solution to this problem. This study evaluates the potential of a large 12-lead ECG dataset – PTB-XL for transfer learning.

## Datasets
  1. PTB-XL: https://physionet.org/content/ptb-xl/1.0.1/
  2. G12EC: https://www.kaggle.com/bjoernjostein/georgia-12leadecg-challenge-database

## Aim
The primary aim of this study is to conduct comprehensive experiments to test whether PTB-XL is a good dataset for transfer learning on smaller ECG datasets. For the main experiment, XResNet model was pre-trained on PTB-XL and the Georgia 12 Lead ECG dataset (G12EC) was used for fine-tuning. A standalone model on the G12EC dataset was also trained for comparison. Further experiments were conducted to evaluate the performance thoroughly:
  1. Reduce the size of evaluation dataset (G12EC) used for fine-tuning the base model and evaluate the performance metric using the rest of the dataset. Also, use the     same train-test split for the standalone model for comparison.
  2. Reduce the size of the base dataset (PTB-XL) for pre-training and evaluate the results by fine-tuning on 10% of the evaluation dataset.
  3. Evaluate the impact of down sampling the original frequency (500Hz) of the two datasets on both the fine-tuned and standalone model.
  4. Visualize the calibration curves across labels and attempt to improve the calibration probabilities.
  5. Transform both ECG datasets into frequency domain and evaluate its impact on pre-training and fine-tuning.

## Findings
For the standalone model trained on G12EC, xresnet1d101 leads to the highest macro AUC score of 0.843 as compared to the remaining deep learning models. The xresnet1d101 architecture was also used for pre-training and fine-tuning as it demonstrated best performance on the PTB-XL dataset. PTB-XL pre-training achieved a macro AUC of 0.947. Consequently, when G12EC was fine-tuned using these pre-trained PTB-XL weights, it led to a macro AUC of 0.919 which is a 0.076 improvement over the model trained independently on G12EC. This improvement can be attributed to the pre-trained weights consisting of fine level ECG signal features which may not be available in the G12EC dataset.

![image](https://user-images.githubusercontent.com/22643549/150191630-0dc19415-5681-4e63-b794-f490b46829bb.png)

Further detailed results and experiments are discussed in the presentation [here](https://github.com/DeepkamalGill/Transfer-Learning-for-ECG/blob/main/ProjectPresentation.pdf).
