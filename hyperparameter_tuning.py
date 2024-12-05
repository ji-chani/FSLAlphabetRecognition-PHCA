# This program seeks for the hyperparameters
# of the classifiers: SVM, RF, LDA, KNN, CART
# that will be used for the main classification

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from modules import prepared_data, plot_metrics

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# global controls
save_figs = False
tune_phca = False
n_trials = 1
n_folds = 5
models = ['svm', 'rf', 'knn', 'lda', 'cart', 'phca']
metrics = ['precision', 'recall', 'f1-score', 'specificity', 'support', 'accuracy']
random_states = [1, 12, 123, 1234, 12345]  # length(random_states) = n_trials

# tuning parameters
svm_params = {'C': stats.uniform(2**-3, 2**15), 
              "gamma": ["auto", "scale"], 
              "kernel": ['rbf']}
rf_params = {"n_estimators": np.arange(1,350+1), 
             "max_depth": np.arange(1,5+1), 
             "min_samples_split": np.arange(1,10+1)}
knn_params = {"n_neighbors": np.arange(1,500+1)}
lda_params = {"solver": ["svd", "lsqr", "eigen"],
                "shrinkage": stats.uniform(0, 1)}
cart_params = {"max_depth":np.arange(1, 30+1),
               "min_samples_leaf": np.arange(1,60+1),
               "min_samples_split": np.arange(1,60+1)}
phca_params = {"dim": [[0], [1], [0,1]]}

best_params_balanced = {mod: [] for mod in models}
best_score_balanced = {mod: [] for mod in models}
best_params_imbalanced = {mod: [] for mod in models}
best_score_imbalanced = {mod: [] for mod in models}

def save_plot(model_base, model_tuned, model_name, X_test, y_test, metrics=metrics, save_figs=save_figs):
        fig = plt.figure(figsize=(11,5))
        plot_metrics(model_base.predict(X_test), y_test, metrics, ax=fig.add_subplot(1,2,1), title="Base")
        plot_metrics(model_tuned.predict(X_test), y_test, metrics, ax=fig.add_subplot(1,2,2), title="Tuned")
        plt.suptitle(model_name)
        if save_figs:
            plt.savefig(f'tuned_models/{model_name}.png')

# ======================================= IMPLEMENTATION PROPER =================================

# Loading the dataset ----------------------------------
print("Loading the dataset ...")
FSL_dataset = np.load(f'FSL_alphabet_landmarks_24classes.npy', allow_pickle=True).item()
num_data = len(FSL_dataset['target'])
print(f'{num_data} data points collected. Each data point represents the MediaPipe landmarks of the hand. \n')

for i in range(n_trials):
    print(f"Running Trial {i+1} --------------------- \n")
    
    # Data preparation --------------------------------------
    
    print('Images not converted into landmarks are removed. \n')
    # balanced dataset
    X_bal, y_bal, paths_bal = prepared_data(FSL_dataset, balanced=True, random_state=21)  # random_state for choosing instances
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

    # splitting dataset (90:10 train-test ratio)
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

    # ---------- Hyperparameter Tuning (Randomized Search Cross Validation)
    print('Tuning hyper-parameters of classifiers using Random Search CV. \n')

    # List of Classifiers
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from modules import PHCA

    estimators = [SVC(), RandomForestClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(), DecisionTreeClassifier()]
    params = [svm_params, rf_params, knn_params, lda_params, cart_params]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_states[i])

    # tuning the conventional classifiers ----
    for idx in range(len(estimators)):
        classifier = models[idx]
        print(f'{classifier} -----------')

        # balanced dataset
        clsf_tuned = RandomizedSearchCV(estimator=estimators[idx],
                            param_distributions=params[idx],
                            n_iter=20,
                            scoring=['accuracy'],
                            refit='accuracy',
                            cv=cv,
                            verbose=1)
        clsf_tuned.fit(X_train_bal, y_train_bal)
        best_params_balanced[classifier].append(clsf_tuned.best_params_)
        best_score_balanced[classifier].append(clsf_tuned.best_score_)
        print(f'Optimal hyperparameters of {classifier} for Balanced dataset are {clsf_tuned.best_params_}')
        print(f"Highest accuracy of optimal {classifier}: {clsf_tuned.best_score_} \n")
        
        # imbalanced dataset
        clsf_tuned = RandomizedSearchCV(estimator=estimators[idx],
                            param_distributions=params[idx],
                            n_iter=20,
                            scoring=['accuracy'],
                            refit='accuracy',
                            cv=cv,
                            verbose=1)
        clsf_tuned.fit(X_train_imb, y_train_imb)
        best_params_imbalanced[classifier].append(clsf_tuned.best_params_)
        best_score_imbalanced[classifier].append(clsf_tuned.best_score_)
        print(f'Optimal hyperparameters of {classifier} for Imbalanced dataset are {clsf_tuned.best_params_}')
        print(f"Highest accuracy of optimal {classifier}: {clsf_tuned.best_score_} \n")

    # tuning PHCA --------
    from sklearn.metrics import accuracy_score
    classifier = "phca"

    # balanced and imbalanced dataset
    cv_splits_bal = [(indices[0], indices[1]) for indices in cv.split(X_train_bal, y_train_bal)]
    cv_splits_imb = [(indices[0], indices[1]) for indices in cv.split(X_train_imb, y_train_imb)]

    if tune_phca:
        print(f'{classifier} -----------')

        # tuning on balanced dataset
        print("Fitting 5 folds for each of 3 candidates, totalling 15 fits")
        phca_score = []
        for k in range(3):  # number of parameters of PHCA
            phca_tuned = PHCA(dim=phca_params["dim"][i])
            accuracy = 0
            for j in range(n_folds): # 5 fold CV
                train_idx, val_idx = cv_splits_bal[j][0], cv_splits_bal[j][1]
                phca_tuned.fit(X_train_bal[train_idx], y_train_bal[train_idx])
                accuracy += accuracy_score(y_train_bal[val_idx], phca_tuned.predict(X_train_bal[val_idx]))
            phca_score.append(accuracy/n_folds)
        
        phca_best_param = phca_params["dim"][np.argmax(phca_score)]
        best_params_balanced[classifier].append(phca_best_param)
        best_score_balanced[classifier].append(max(phca_score))
        print(f'Optimal hyperparameters of {classifier} for Balanced dataset are dim: {phca_best_param}')
        print(f"Highest accuracy of optimal {classifier}: {max(phca_score)} \n")

        # tuning on imbalanced dataset
        print("Fitting 5 folds for each of 3 candidates, totalling 15 fits")
        phca_score = []
        for k in range(3):  # number of parameters of PHCA
            phca_tuned = PHCA(dim=phca_params["dim"][i])
            accuracy = 0
            for j in range(n_folds): # 5 fold CV
                train_idx, val_idx = cv_splits_imb[j][0], cv_splits_imb[j][1]
                phca_tuned.fit(X_train_imb[train_idx], y_train_imb[train_idx])
                accuracy += accuracy_score(y_train_imb[val_idx], phca_tuned.predict(X_train_imb[val_idx]))
            phca_score.append(accuracy/n_folds)
        
        phca_best_param = phca_params["dim"][np.argmax(phca_score)]
        best_params_imbalanced[classifier].append(phca_best_param)
        best_score_imbalanced[classifier].append(max(phca_score))
        print(f'Optimal hyperparameters of {classifier} for Imbalanced dataset are dim: {phca_best_param}')
        print(f"Highest accuracy of optimal {classifier}: {max(phca_score)} \n")
    else:
        # from experimentations, maxdim=0 is always best parameter for PHCA
        phca_best_param = [0]
        best_params_balanced[classifier].append(phca_best_param)
        best_params_imbalanced[classifier].extend(phca_best_param)

        # balanced ---- obtaining (best) average accuracy/score
        accuracy = 0
        for j in range(n_folds):
            train_idx, val_idx = cv_splits_bal[j][0], cv_splits_bal[j][1]
            phca_tuned = PHCA(dim=phca_best_param[j])
            phca_tuned.fit(X_train_bal[train_idx], y_train_bal[train_idx])
            accuracy += accuracy_score(y_train_bal[val_idx], phca_tuned.predict(X_train_bal[val_idx]))
        best_score_balanced[classifier].append(accuracy/n_folds)
        print(f'Optimal hyperparameters of {classifier} for Balanced dataset are dim: {phca_best_param}')
        print(f"Highest accuracy of optimal {classifier}: {accuracy/n_folds} \n")

        # imbalanced ---- obtaining (best) average accuracy/score
        accuracy = 0
        for j in range(n_folds):
            train_idx, val_idx = cv_splits_bal[j][0], cv_splits_bal[j][1]
            phca_tuned = PHCA(dim=phca_best_param[j])
            phca_tuned.fit(X_train_imb[train_idx], y_train_imb[train_idx])
            accuracy += accuracy_score(y_train_imb[val_idx], phca_tuned.predict(X_train_imb[val_idx]))
        best_score_imbalanced[classifier].append(accuracy/n_folds)
        print(f'Optimal hyperparameters of {classifier} for Imbalanced dataset are dim: {phca_best_param}')
        print(f"Highest accuracy of optimal {classifier}: {accuracy/n_folds} \n")
            

# save best parameters as txt files
def dict_to_txt(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

dict_to_txt(best_params_balanced, f"best_params_score/best_params_balanced_{n_trials}trials")
dict_to_txt(best_params_imbalanced, f"best_params_score/best_params_imbalanced_{n_trials}trials")
dict_to_txt(best_score_balanced, f"best_params_score/best_score_balanced_{n_trials}trials")
dict_to_txt(best_score_imbalanced, f"best_params_score/best_score_imbalanced_{n_trials}trials")
