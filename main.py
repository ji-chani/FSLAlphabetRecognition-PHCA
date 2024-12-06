# Static FSL Alphabet Recognition using PHCA

# classifiers are used with best tuned hyperparameters
# classifiers are trained using train data
# classifiers are evaluated using test data

__author___ = "CBJetomo"

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from modules import FilipinoSignLanguage, Image2Landmarks, PHCA, prepared_data, compare_predictions

# ------------ Global controls
collect_data = False # True when landmarks data are still not extracted
save_figs = True
n_trials = 1

# ------------ Global parameters
classes = 24  # static
models = ['svm', 'rf', 'knn', 'lda', 'cart', 'phca']
metrics = ['precision', 'recall', 'f1-score', 'specificity', 'support', 'accuracy']
zip_path = 'FSL_images_static.zip'
random_states = [1, 12, 123, 1234, 12345]

# ------------ Dataset Collection

print('Collecting the dataset ...')
if collect_data:
    ## Collecting Image Paths
    fsl_alphabet = FilipinoSignLanguage(zip_path, data_extension='jpg')
    dataset_paths = fsl_alphabet.load_fsl(classes)

    ## Collecting Landmarks using MediaPipe
    num_data = len(dataset_paths['target'])
    FSL_dataset = {'data': [], 'target': [], 'path': []}

    img2lmarks = Image2Landmarks(flatten=True, display_image=False)
    for _, n in enumerate(tqdm(np.arange(num_data))):
        image_path = dataset_paths['data'][n]
        hand_landmarks = img2lmarks.image_to_hand_landmarks(image_path)
        FSL_dataset['data'].append(hand_landmarks)
        FSL_dataset['target'].append(dataset_paths['target'][n])
        FSL_dataset['path'].append(dataset_paths['data'][n])
        
    np.save(f'FSL_alphabet_landmarks_{classes}classes.npy', FSL_dataset)

FSL_dataset = np.load(f'FSL_alphabet_landmarks_{classes}classes.npy', allow_pickle=True).item()
num_data = len(FSL_dataset['target'])
print(f'{num_data} data points collected. Each datapoint represents the MediaPipe landmarks of the hand.')

predicted_labels_balanced = {mod: [] for mod in models+['true_labels']}
predicted_labels_imbalanced = {mod: [] for mod in models+['true_labels']}

# obtain best hyperparameters
def txt_to_dict(filename):
    with open(filename, 'r') as file:
        dictionary = {}
        for line in file:
            key, value = line.strip().split(': ', 1)
            # Try to parse lists or numerical values
            try:
                value = eval(value)
            except (NameError, SyntaxError):
                pass  # Value remains a string if eval fails
            dictionary[key] = value
    return dictionary

best_params_balanced = txt_to_dict(f"best_params_score/best_params_balanced.txt")
best_params_imbalanced = txt_to_dict(f"best_params_score/best_params_imbalanced.txt")

for i in range(n_trials):
    print(f"Running Trial {i+1} ========================== \n")

    # ------------ Data Preparation

    print("Images not converted into landmarks are removed. \n")

    # balanced dataset
    X_bal, y_bal, paths_bal = prepared_data(FSL_dataset, balanced=True, random_state=random_states[i]) # randomly chooses instances
    print(f"""Balanced dataset prepared.
        total num_instances                 : {X_bal.shape[0]}
        num_instances per class             : {X_bal.shape[0]/24:n}
        num_features                        : {X_bal.shape[1]}""")
    
    # imbalanced dataset
    X_imb, y_imb, paths_imb = prepared_data(FSL_dataset, balanced=False)
    freqs = np.unique(y_imb, return_counts=True)[1]
    print(f"""Imbalanced dataset prepared.
        total num_instances                 : {X_imb.shape[0]}
        minimum num_instances per class     : {min(freqs):n}
        maximum num_instances per class     : {max(freqs):n}
        average num_instances per class     : {np.mean(freqs):0.2f}
        num_features                        : {X_imb.shape[1]} \n""")
    
    # splitting dataset (90:20 train-test ratio)
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal, y_bal, test_size=0.10, stratify=y_bal, random_state=random_states[i])
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(X_imb, y_imb, test_size=0.10, stratify=y_imb, random_state=random_states[i])
    print("Balanced and Imbalanced datasets are split into train (90%) and test (10%) sets, respectively. \n")

    # scaling dataset
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test_bal = scaler.transform(X_test_bal)

    X_train_imb = scaler.fit_transform(X_train_imb)
    X_test_imb = scaler.transform(X_test_imb)
    print("Datasets are scaled using Standard Scaler. \n")
    
    # ------------- Classification of Test Set (using optimal hyperparameters)
    print("Classifying test data using tuned classifiers ============= \n")

    estimators = [SVC(), RandomForestClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(), DecisionTreeClassifier(), PHCA()]
    params_bal = [best_params_balanced[mod][i] for mod in models]
    params_imb = [best_params_imbalanced[mod][i] for mod in models]

    for idx in range(len(estimators)):
        classifier = models[idx]
        print(f'{classifier} ---------------------')

        # balanced dataset -----
        # pass optimal hyperparameters to classifier
        clsf_tuned = estimators[idx].set_params(**params_bal[idx])
        
        print("(1) learning from the Balanced dataset ...")
        clsf_tuned.fit(X_train_bal, y_train_bal)  # training
        print("(2) finished learning.")
        print("(3) predicting new data ...")
        predicted_labels_balanced[classifier].append(clsf_tuned.predict(X_test_bal))  # testing
        print("(4) finished predicting the Balanced dataset. \n")

        # imbalanced dataset -----
        clsf_tuned = estimators[idx].set_params(**params_imb[idx])
        
        print("(1) learning from the Imbalanced dataset ...")
        clsf_tuned.fit(X_train_imb, y_train_imb)  # training
        print("(2) finished learning.")
        print("(3) predicting new data ...")
        predicted_labels_imbalanced[classifier].append(clsf_tuned.predict(X_test_imb))  # testing
        print("(4) finished predicting the Imbalanced dataset. \n\n")
        

    # save correct classes
    predicted_labels_balanced['true_labels'].append(y_test_bal)
    predicted_labels_imbalanced['true_labels'].append(y_test_imb)

    # save predictions
    np.save(f'predicted_labels_balanced.npy', predicted_labels_balanced)
    np.save(f'predicted_labels_imbalanced.npy', predicted_labels_imbalanced)

    # Classification Report
    print("Results for the Balanced Dataset =================")
    for mod in models:
        print(f'{mod} ---------------------')
        print(classification_report(predicted_labels_balanced['true_labels'][i], predicted_labels_balanced[mod][i]))

    print("Results for the Imbalanced Dataset =================")
    for mod in models:
        print(f'{mod} ---------------------')
        print(classification_report(predicted_labels_imbalanced['true_labels'][i], predicted_labels_imbalanced[mod][i]))


# save trained models
# import pickle
# for idx, mod in enumerate(trained_models):
#     pickle.dump(mod, open(f'trained_{models[idx]}x`_{classes}classes.sav', 'wb'))
# pickle.dump(scaler, open(f'trained_scaler_{classes}classes.sav', 'wb'))

# for met in metrics:
#     if met != 'support':
#         fig = plt.figure()
#         plot_metrics_per_metric(predicted_labels, met, save=save_results)

# compare_predictions(predicted_labels, metrics)
