from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_curve, auc, f1_score, \
    make_scorer
from utils import elbow_method
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def find_best_n_components_pca():
    data = np.load('./data/parameters/X_y_10000_norm.npz')
    X, y = data['arr_0'], data['arr_1']
    # Compute the mean of each column
    X_mean = X.mean(axis=0)
    # Save the preprocessing mean
    pickle.dump(X_mean, open('./models/mean_preprocess.pickle', 'wb'))
    # Scale the data
    X = X - X_mean

    # We want to keep all components but still want to gain in time computing so we use svd_solver='auto' to get
    # n_components equal to the minimum of the dimensions of X
    pca = PCA(n_components=None, whiten=True, svd_solver='auto', random_state=0)
    # It may take a long time to apply PCA on X data
    X_pca = pca.fit_transform(X)
    eigen_ratio = pca.explained_variance_ratio_
    eigen_ratio_cum = np.cumsum(eigen_ratio)

    # Visualize eigen_ratio and eigen_ratio_cumsum plots to find the best no. of components
    elbow_method(eigen_ratio, eigen_ratio_cum, 200)


def train_pca(n_components=50):
    data = np.load('./data/parameters/X_y_10000_norm.npz')
    X, y = data['arr_0'], data['arr_1']

    pca = PCA(n_components=n_components, whiten=True, svd_solver='auto', random_state=0)
    X_pca = pca.fit_transform(X)
    # Save pca model
    pickle.dump(pca, open(f'./models/pca_{n_components}.pickle', 'wb'))
    # Save new X_pca
    np.savez(f'./data/parameters/X_pca_{n_components}_y_norm.npz', X_pca, y)


def train_evaluate_svm(C=5, kernel='rbf', gamma=0.01, save=True):
    data = np.load('./data/parameters/X_pca_50_y_norm.npz')
    X, y = data['arr_0'], data['arr_1']
    # Split data into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=0)
    model.fit(X_train, y_train)

    # Model evaluation
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = np.concatenate((cm, cm.sum(axis=0).reshape(1,-1)), axis=0)
    cm = np.concatenate((cm, cm.sum(axis=1).reshape(-1, 1)), axis=1)
    plt.imshow(cm)
    for i in range(3):
        for j in range(3):
            plt.text(i, j, '%d'%cm[i,j])
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.savefig('./report/confusion_matrix.png')

    # Classification report
    cr = classification_report(y_test, y_pred, target_names=['male', 'female'], output_dict=True)
    kappa_score = cohen_kappa_score(y_test, y_pred)

    # Plot ROC Curve
    fpr, tpr, threshold = roc_curve(y_test, y_prob[:, 1])
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, '-.')
    plt.plot([0, 1], [0, 1], 'b--')
    for i in range(0, len(threshold), 20):
        plt.plot(fpr[i], tpr[i], '^')
        plt.text(fpr[i], tpr[i], "%0.2f" % threshold[i])
    plt.legend(['AUC Score = %0.2f' % auc_score])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.savefig('./report/roc_curve.png')

    # Save all metrics in a csv file
    metrics = pd.DataFrame(cr)
    metrics.drop(['accuracy'], axis=1, inplace=True)
    metrics['train_accuracy'] = train_accuracy
    metrics['test_accuracy'] = test_accuracy
    metrics['kappa'] = kappa_score
    metrics['auc'] = auc_score
    metrics.to_csv('./report/metrics.csv', sep=";", header=True, index=True)

    # Save SVM model
    if save:
        pickle.dump(model, open(f'./models/svm_{C}_{kernel}_{gamma}.pickle', 'wb'))


def tune_svm():
    data = np.load('./data/parameters/X_pca_50_y_norm.npz')
    X, y = data['arr_0'], data['arr_1']
    # Split data into train set and test set
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    model_tune = SVC(random_state=0)
    f1_macro = make_scorer(f1_score, pos_label=None, average="macro")
    # Params Grid
    param_grid = {
        'C': [1, 5, 10, 20, 30],
        'kernel': ['rbf', 'poly'],
        'gamma': [0.1, 0.01, 0.001],
        'coef0': [0, 1]
    }

    # GridSearch CV hyperparameters tuning
    cv = StratifiedKFold(n_splits=5, random_state=0)
    model_grid = GridSearchCV(model_tune, param_grid=param_grid, scoring=f1_macro, cv=cv, verbose=1)
    model_grid.fit(X_train, y_train)

    # Save best_params in a csv file
    best_params = model_grid.best_params_
    best_params['best_score'] = model_grid.best_score_
    df_best_params = pd.DataFrame(best_params, index=['best params'])
    df_best_params.to_csv('./report/best_params.csv', sep=";", header=True, index=True)




