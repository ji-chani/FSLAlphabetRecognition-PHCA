from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import scikit_posthocs as sp
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import seaborn as sns

from modules import Image2Landmarks

def kfoldDivideData(FSLData, FSLTarget, folds=5):
    """ Partitions the dataset into k folds, each fold with equal number of data from the same class."""
    fivefoldData, fivefoldTarget = [[] for i in range(folds)], [[] for i in range(folds)]
    feat, lab = [], []
    fold = 0
    num_data, num_target = len(FSLData), len(np.unique(FSLTarget))
    dpcpf = math.ceil((num_data/num_target)/folds)
    cut = dpcpf

    for i in range(len(FSLTarget)):
        if i < cut:
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
        else:
            fivefoldData[fold] += feat
            fivefoldTarget[fold] += lab

            feat, lab = [], []
            feat.append(FSLData[i]), lab.append(FSLTarget[i])
            fold += 1
            cut += dpcpf

        if fold == folds:
            fold = 0
    fivefoldData[folds-1] += feat
    fivefoldTarget[folds-1] += lab

    for f in range(folds):
        fivefoldData[f], fivefoldTarget[f] = shuffle(fivefoldData[f], fivefoldTarget[f], random_state=np.random.randint(folds))
    return fivefoldData, fivefoldTarget

def balance_data(dataset):
    balanced_dataset = {'data': [], 'target': [], 'path': []}
    # removing None data values
    rem_idx = []
    for idx, data in enumerate(dataset['data']):
        if data is None:
            rem_idx.append(idx)

    rem_idx.sort(reverse=True)
    for idx in rem_idx:
        dataset['data'].pop(idx)
        dataset['target'].pop(idx)
        dataset['path'].pop(idx)

    # balancing num of data per class according to minimum
    classes = np.unique(dataset['target'])
    numdata_per_class = min([len(np.where(np.array(dataset['target']) == c)[0]) for c in classes])
    keep_idx = []
    for c in classes:
        keep = np.where(np.array(dataset['target']) == c)[0][:numdata_per_class]
        keep_idx.extend(keep)
    balanced_dataset['data'] = [dataset['data'][idx] for idx in keep_idx]
    balanced_dataset['target'] = [dataset['target'][idx] for idx in keep_idx]
    balanced_dataset['path'] = [dataset['path'][idx] for idx in keep_idx]
    return np.array(balanced_dataset['data']), np.array(balanced_dataset['target']), np.array(balanced_dataset['path'])

def get_specificity(confusionMatrix, classes):
    label_lists = classes
    specificity = {}
    for l, label in enumerate(label_lists):
        tp, tn, fp, fn = 0, 0, 0, 0
        tp = confusionMatrix[l, l]
        fn = sum(confusionMatrix[l]) - tp
        for i in range(len(label_lists)):
            for j in range(len(label_lists)):
                if i == l or j == l:
                    continue
                else:
                    tn += confusionMatrix[i,j]
        for i in range(len(label_lists)):
            if i==l:
                continue
            else:
                fp += confusionMatrix[l][i]
        specificity[str(label)] = tn/(tn+fp)
    return specificity

def classification_report_with_specificity(predicted_labels):
    report = {key: classification_report(predicted_labels['true_labels'], predicted_labels[key], output_dict=True) for key in predicted_labels.keys() if key != 'true_labels'}
    for key in report.keys():
        cf = confusion_matrix(predicted_labels['true_labels'], predicted_labels[key], labels=np.unique(predicted_labels['true_labels']))
        specificity = get_specificity(cf, classes=np.unique(predicted_labels['true_labels']))
     
        avg = 0
        for label in specificity.keys():
            avg += specificity[label]
            report[key][label]['specificity'] = specificity[label]
    
    report[key]['macro avg']['specificity'] = avg/len(specificity)
    report[key]['weighted avg']['specificity'] = avg/len(specificity)
    return report

def plot_metrics_per_metric(predicted_labels:dict, metric:str, save:bool=False, ax=None):
    colors = ['#334085','#286c8b','#1ba394','#17a88c','#0eda9b', '#68f0c0']
    report = classification_report_with_specificity(predicted_labels)
    models = [mod for mod in predicted_labels.keys() if mod != 'true_labels']  # extract models
    labels = [lab for lab in report['phca'].keys() if len(lab) == 1]  # extract classes
    
    # extracting values
    if metric == 'accuracy':
        ave_metric_values = [report[mod]['accuracy'] for mod in models]
        ylabel = 'overall ' + metric
    else:
        metric_values = []
        for mod in models:
            metric_values.append([report[mod][lab][metric] for lab in labels])
        ave_metric_values = [np.mean(mv) for mv in metric_values]
        ylabel = 'ave. ' + metric

    # plotting
    ax = ax or plt.gca()
    b = ax.bar(np.arange(len(models)), ave_metric_values, color=colors)
    ax.bar_label(b, fmt='%.2f', label_type='center')
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylabel(ylabel, fontsize=15)
    sns.despine()
    if save:
        plt.savefig(f'final_results/result_{metric}.png')

def nemenyi_test(predicted_labels, metrics, alpha=0.05):
    report = classification_report_with_specificity(predicted_labels)
    label_list = np.unique(predicted_labels['true_labels'])
    report_list = {}
    models = [mod for mod in report.keys() if mod != 'true_labels']
    for model in models:
        report_list[model] = []
        for met in metrics:
            if met == 'accuracy':
                report_list[model].append(report[model][met])
            elif met == 'support':
                continue
            else:
                for label in label_list:
                    report_list[model].append(report[model][str(label)][met])
    stat, pvalue = stats.friedmanchisquare(*[report_list[mod] for mod in models])
    print(f'The p-value is: {pvalue}')
    if pvalue <= alpha:
        print('At least one population mean differs from the others.')
    else:
        print('The mean value for each of the population is equal.')
    data = np.array([report_list[mod] for mod in models])
    nemenyi_table = sp.posthoc_nemenyi_friedman(data.T)
    nemenyi_table = nemenyi_table.set_axis(models, axis=1)
    nemenyi_table = nemenyi_table.set_axis(models, axis=0)
    print(nemenyi_table)
    return nemenyi_table

def mcnemar_test(predicted_labels, alpha=0.05):
    mcnemar_test = {}
    models = [mod for mod in predicted_labels.keys() if mod != 'true_labels']
    true_labels = predicted_labels['true_labels']
    for mod in models:
        if mod == 'phca':
            continue
        mcnemar_matrix = np.zeros((2,2))
        for i in range(len(true_labels)):
            if predicted_labels['phca'][i] == true_labels[i]:  # correct predictions of PHCA
                if predicted_labels[mod][i] == true_labels[i]:  # correct PHCA and model
                    mcnemar_matrix[0][0] += 1
                else:  # correct PHCA incorrect model
                    mcnemar_matrix[0][1] += 1
            else:  # incorrect predictions of PHCA
                if predicted_labels[mod][i] == true_labels[i]:  # incorrect PHCA, correct model
                    mcnemar_matrix[1][0] += 1
                else:  # incorrect PHCA, incorrect model
                    mcnemar_matrix[1][1] += 1
        pvalue = mcnemar(mcnemar_matrix, exact=False, correction=False).pvalue
        print(f'The pvalue is: {pvalue}')
        if pvalue <= alpha:
            if mcnemar_matrix[0][1]>=mcnemar_matrix[1][0]:
                print(f'The PHCA model performs better than {mod} with ratio {int(mcnemar_matrix[0][1])}:{int(mcnemar_matrix[1][0])}')
            else:
                print(f'The PHCA model performs worse than {mod} with ratio {int(mcnemar_matrix[0][1])}:{int(mcnemar_matrix[1][0])}')
        else:
            print(f'The performance of the PHCA model and {mod} are equal.')

        mcnemar_test[mod] = {'value': pvalue, 'matrix': mcnemar_matrix}

    return mcnemar_test

def compare_predictions(predicted_labels, metrics):
    nemenyi_result = nemenyi_test(predicted_labels, metrics)
    mcnemar_result = mcnemar_test(predicted_labels)

    return nemenyi_result, mcnemar_result
