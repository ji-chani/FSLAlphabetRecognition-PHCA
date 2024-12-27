from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

import scikit_posthocs as sp
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns

import numpy as np
import cv2
import math

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

def prepared_data(dataset:dict, balanced:bool=True, random_state:int=None):
    """Returns dataset with `balanced=True` or `balanced=False` number of instances per class."""
    
    np.random.seed(random_state)
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

    if balanced: # balancing num of data per class according to minimum (313 instances)
        classes = np.unique(dataset['target'])
        numdata_per_class = min([len(np.where(np.array(dataset['target']) == c)[0]) for c in classes])
        keep_idx = []
        for c in classes: # randomly selects indices
            clss_indices = np.where(np.array(dataset['target']) == c)[0]
            keep = np.random.choice(clss_indices, size=numdata_per_class, replace=False)
            keep_idx.extend(np.sort(keep))
        balanced_dataset['data'] = [dataset['data'][idx] for idx in keep_idx]
        balanced_dataset['target'] = [dataset['target'][idx] for idx in keep_idx]
        balanced_dataset['path'] = [dataset['path'][idx] for idx in keep_idx]
        return np.array(balanced_dataset['data']), np.array(balanced_dataset['target']), np.array(balanced_dataset['path'])
    else:
        return np.array(dataset['data']), np.array(dataset['target']), np.array(dataset['path'])

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

def classification_report_with_specificity(predicted_labels:dict, idx:int):
    """ Include specificity to classification report """
    report = {key: classification_report(predicted_labels['true_labels'][idx], predicted_labels[key][idx], output_dict=True) for key in predicted_labels.keys() if key != 'true_labels'}
    for key in report.keys():
        cf = confusion_matrix(predicted_labels['true_labels'][idx], predicted_labels[key][idx], labels=np.unique(predicted_labels['true_labels'][idx]))
        specificity = get_specificity(cf, classes=np.unique(predicted_labels['true_labels'][idx]))
     
        avg = 0
        for label in specificity.keys():
            avg += specificity[label]
            report[key][label]['specificity'] = specificity[label]
    
    report[key]['macro avg']['specificity'] = avg/len(specificity)
    report[key]['weighted avg']['specificity'] = avg/len(specificity)
    return report

def plot_boxplots(predicted_labels:dict, metric:str, save_fig:bool=False, ax=None):
    """
    Plot boxplots for each `metric`
    
    ----------
    :param: predicted_labels: dictionary with classifiers (keys) and obtained results per class (values)
    :param: metric: single scoring metrics for comparison of classifiers `['precision', 'recall', 'f1-score', 'specificity', 'support', 'accuracy']`
    :param: save: Boolean on whether to save plot or not
    :param: ax: can be used to subplot result

    :return: average score of `classifier` per `metric` across `n_trials`
    """
    # color per classifier
    colors = ['#334085','#286c8b','#1ba394','#17a88c','#0eda9b', '#68f0c0']
    models = [mod for mod in predicted_labels.keys() if mod != 'true_labels']  # extract models
    labels = np.unique(predicted_labels['true_labels'])  # extract classes (labels)
    n_trials = len(predicted_labels['true_labels'])

    scores = []
    for i in range(n_trials):
        report = classification_report_with_specificity(predicted_labels, idx=i)
        
        # extracting values
        if metric == 'accuracy':
            ave_metric_values = [report[mod]['accuracy'] for mod in models]
            ylabel = 'Overall ' + metric.capitalize()
        else:
            metric_values = []
            for mod in models:
                metric_values.append([report[mod][lab][metric] for lab in labels])
            ave_metric_values = [np.mean(mv) for mv in metric_values]
            ylabel = 'Average ' + metric.capitalize()
        
        scores.append(ave_metric_values)
    scores = np.array(scores)

    # plotting
    ax = ax or plt.gca()
    bplot = ax.boxplot(scores, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):  # change color of bplots
        patch.set_facecolor(color)

    # plot adjustments
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Classifiers")
    ax.grid(axis='y')

    sns.despine()
    if save_fig:
        plt.savefig(f'figures/result_{metric}.png')
    return scores

def plot_bars(ave_scores:dict, metrics:list, save_fig:bool=False, ax=None):
    """ 
    Plot grouped bar graphs of average scores of classifers acc to all `metrics`
    
    -----------
    :param: predicted_labels: dictionary with classifiers (keys) and obtained results per class (values)
    :param: metrics: list of all metrics 
    :param: save_fig: Boolean on whether to save plot or not
    :param: ax: can be used to subplot result
    """
    colors = ['#334085','#286c8b','#1ba394','#0eda9b', '#68f0c0']
    models = ['svm', 'rf', 'knn', 'lda', 'cart', 'phca']
    
    # for grouping bars
    width = 0.25
    shift = -2 * width
    x = np.arange(len(ave_scores['accuracy']))*1.5

    # bar graphs
    ax = ax or plt.gca()
    for met, col in zip(metrics, colors):
        ax.bar(x+shift, ave_scores[met], width=width, color=col, label=met, edgecolor='k')
        shift += width

    # plot adjustments
    ax.set_xticks(x, models)
    ax.set_ylim(top=1.1)
    ax.legend(loc='lower right')
    ax.set_xlabel("Classifiers")
    ax.set_ylabel("Average values")

    sns.despine()
    if save_fig:
        plt.savefig(f'figures/result_bars.png')

def plot_confusion_matrix(predicted_labels:dict, model:str, idx:int, with_colorbar:bool=False, save_fig:bool=False, title:str=None, ax=None):
    """
    Plot confusion matrix obtained by `model` in Trial `idx+1`

    -----------
    :param: predicted_labels: dictionary with classifiers (keys) and obtained results per class (values)
    :param: model: model to be assessed
    :param: idx: Trial idx
    :param: save_fig: Boolean on whether to save plot or not
    :param: ax: can be used to subplot result
    """
    
    cf = confusion_matrix(predicted_labels['true_labels'][idx], predicted_labels[model][idx])
    classes = np.unique(predicted_labels['true_labels'][0])

    # color palette
    cmp = mpl.colormaps['viridis']
    cmp = ListedColormap(cmp(np.linspace(0.25, 0.75, 128)))
    
    # plot confusion matrix and colorbar
    ax = ax or plt.gca()
    im = ax.imshow(cf, cmap=cmp)
    if with_colorbar:
        plt.colorbar(im)

    # change xticks to classes in the dataset
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # x and y label, title
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title, fontsize=13)

    # create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = 'w'
            if i == j:
                text_color = 'k'
            if cf[i,j] != 0:
                ax.text(j, i, cf[i,j], ha="center", va="center", color=text_color)

    if save_fig:
        plt.savefig(f'figures/result_confusion.png')

def plot_predictions(y_pred, y_test, test_ind, num_images, landmarks:bool, title, img_paths):
  if title == 'Correct':
    indices = test_ind[np.nonzero(y_pred == y_test)[0]]
  else:
    indices = test_ind[np.nonzero(y_pred != y_test)[0]]
  num_axs = int(np.sqrt(num_images))


  plt.figure(figsize=(10,10))
  for i, correct, in enumerate(indices[:num_images]):
    plt.subplot(num_axs, num_axs, i+1)
    if title == "Correct":
      pred_ind = np.nonzero(y_pred == y_test)[0][i]
    else:
      pred_ind = np.nonzero(y_pred != y_test)[0][i]

    if landmarks:
      img2landmarks = Image2Landmarks(flatten=False, display_image=True)
      _ = img2landmarks.image_to_hand_landmarks(img_paths[correct])
    else:
      img = cv2.imread(img_paths[correct])
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted {y_pred[pred_ind]}, Class {y_test[pred_ind]}")
    plt.axis('off')
  plt.suptitle(f'{title} Predictions')
  plt.tight_layout()
  plt.show()

def nemenyi_test(predicted_labels, idx, metrics, alpha=0.05):
    report = classification_report_with_specificity(predicted_labels, idx)
    label_list = np.unique(predicted_labels['true_labels'][idx])
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
    _, pvalue = stats.friedmanchisquare(*[report_list[mod] for mod in models])
    print(f'The p-value is: {pvalue}')
    if pvalue <= alpha:
        print('At least one population mean differs from the others. \n')
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
