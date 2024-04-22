# Classification of Static FSL Alphabet using PHCA

import numpy as np
from tqdm import tqdm
from modules import FilipinoSignLanguage, Image2Landmarks, PHCA, kfoldDivideData, balance_data, plot_metrics_per_metric, compare_predictions

# ------------ Parameters
classes = 24  # static
zip_path = 'FSL_images_static.zip'
metrics = ['precision', 'recall', 'f1-score', 'specificity', 'support', 'accuracy']
collect_data = True  # True when landmarks data are still not extracted
save_results = True


# ------------ Dataset Collection

print('Collecting data ...')
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


# ------------ Data Preparation

## balancing data (removing None values)
X, y, FSL_path = balance_data(FSL_dataset)
print(f'None landmarks values removed.')
print(f'The data consists of {X.shape[0]} datapoints with {X.shape[1]} features each.')
print(f'That is, we have {int(X.shape[0]/classes)} datapoints per class. \n')

# ------------ Validation (5-fold cross validation)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier


models = ['knn', 'rf', 'svm', 'lda', 'cart', 'phca', 'true_labels']
predicted_labels = {mod: [] for mod in models}

folds = 5
fivefold_X, fivefold_y = kfoldDivideData(X, y, folds=folds)
print('Starting five-fold cross validation --------------- \n')
for val in range(folds):
    print(f'Running validation {val} ... \n')

    # Data Splitting
    X_train, y_train, X_test, y_test = [], [], [], []
    for j in range(folds):
        if j == val:
            X_test.extend(fivefold_X[j])
            y_test.extend(fivefold_y[j])
        else:
            X_train.extend(fivefold_X[j])
            y_train.extend(fivefold_y[j])

    # Data Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Classification
    print('The models are learning from the data ...')

    ## KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # RF
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # SVM
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # CART
    cart = DecisionTreeClassifier()
    cart.fit(X_train, y_train)
    
    # PHCA
    phca = PHCA(dim=0)
    phca.fit(X_train, y_train)

    print('The models are finished learning.')
    print('The models are now predicting new data ...')
    trained_models = [knn, rf, svm_model, lda, cart, phca]
    predicted_labels['true_labels'].extend(y_test)
    for idx, mod in enumerate(trained_models):
        predicted_labels[models[idx]].extend(mod.predict(X_test))
    print('The models are finished predicting. \n')

# save predictions
np.save(f'predicted_labels_{classes}classes.npy', predicted_labels)

# save trained models
import pickle
for idx, mod in enumerate(trained_models):
    pickle.dump(mod, open(f'trained_{models[idx]}_{classes}classes.sav', 'wb'))
pickle.dump(scaler, open(f'trained_scaler_{classes}classes.sav', 'wb'))


# ----------- Classification Report
from sklearn.metrics import classification_report
for mod in models:
    if mod != 'true_labels':
        print(f'{mod} ------------------')
        print(classification_report(predicted_labels['true_labels'], predicted_labels[mod]))

# Plotting
import matplotlib.pyplot as plt

for met in metrics:
    if met != 'support':
        fig = plt.figure()
        plot_metrics_per_metric(predicted_labels, met, save=save_results)

compare_predictions(predicted_labels, metrics)
