from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np
import argparse
import csv
import random
from scipy import stats


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''

    return sum(np.diagonal(C)) / sum(sum(C))


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''

    result = []
    for i in range(len(C)):
        result.append(C[i][i] / sum(C[i]))

    return result


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    for i in range(len(C)):
        result.append(C[i][i] / sum(np.transpose(C)[i]))

    return result


def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    data = np.load(filename)["arr_0"]

    X = []
    y = []

    random.shuffle(data)
    for d in data:
        X.append(d[0:173])
        y.append(d[173])

    # splits data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    classifiers = [SVC(kernel='linear', max_iter=1000),
                   SVC(gamma=2, max_iter=1000),
                   RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05),
                   AdaBoostClassifier()]

    accuracy_list = []
    recall_list = []
    precision_list = []
    cm_list = []

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        c_m = confusion_matrix(y_test, prediction)
        cm_list.append(c_m)
        accuracy_list.append(accuracy(c_m))
        recall_list.append(recall(c_m))
        precision_list.append(precision(c_m))

    with open('a1_3.1.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')

        for i in range(len(accuracy_list)):
            spamwriter.writerow(
                [i + 1] + [accuracy_list[i]] + recall_list[i] + precision_list[i] + cm_list[i].ravel().tolist())

    iBest = np.argmax(accuracy_list) + 1

    return (X_train, X_test, y_train, y_test, iBest)


part2_accuracy_list_1000292033 = []


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    iBest = iBest - 1

    classifiers = [SVC(kernel='linear', max_iter=1000),
                   SVC(gamma=2, max_iter=1000),
                   RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05),
                   AdaBoostClassifier()]

    classifier = classifiers[iBest]
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    c_m = confusion_matrix(y_test, prediction)
    part2_accuracy_list_1000292033.append(accuracy(c_m))

    return (X_train[0:1000], y_train[0:1000])


part3_feature_list_1000292033 = []
part3_accuracy_list_1000292033 = []


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    i = i - 1

    classifiers = [SVC(kernel='linear', max_iter=1000),
                   SVC(gamma=2, max_iter=1000),
                   RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05),
                   AdaBoostClassifier()]

    for k in [5, 10, 20, 30, 40, 50]:

        # experienment with differnet k size
        selector = SelectKBest(f_classif, k)
        selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_
        # print(pp[selector.get_support()])

        # only need record 32k data once
        if len(part3_feature_list_1000292033) < 6:
            selector = SelectKBest(f_classif, k)
            selector.fit_transform(X_train, y_train)
            pp = selector.pvalues_
            list = []
            list.append(k)
            list.extend(pp[selector.get_support()])
            part3_feature_list_1000292033.append(list)

        # train data for part3.2

        if k is 5:
            classifier = clone(classifiers[i])
            classifier.fit(X_1k, y_1k)
            prediction = classifier.predict(X_test)
            c_m = confusion_matrix(y_test, prediction)
            part3_accuracy_list_1000292033.append(accuracy(c_m))

            # adding the 32k data size at the end
            if len(part3_accuracy_list_1000292033) is 5:
                classifier = clone(classifiers[i])
                classifier.fit(X_train, y_train)
                prediction = classifier.predict(X_test)
                c_m = confusion_matrix(y_test, prediction)
                part3_accuracy_list_1000292033.append(accuracy(c_m))


def class34(filename, i):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    i = i - 1
    data = np.load(filename)["arr_0"]

    X = []
    y = []

    for d in data:
        X.append(d[0:173])
        y.append(d[173])

    X = np.array(X)
    y = np.array(y)

    classifiers = [SVC(kernel='linear', max_iter=1000),
                   SVC(gamma=2, max_iter=1000),
                   RandomForestClassifier(max_depth=5, n_estimators=10),
                   MLPClassifier(alpha=0.05),
                   AdaBoostClassifier()]

    kf = KFold(n_splits=5, shuffle=True)

    # global list to store result
    fold_test_result_list = []
    p_values = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        accuracy_list = []
        for clf in classifiers:
            classifier = clone(clf)

            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_test)
            c_m = confusion_matrix(y_test, prediction)
            accuracy_list.append(accuracy(c_m))

        fold_test_result_list.append(accuracy_list)

    vertical_result = np.transpose(fold_test_result_list)

    # compare the result with the best classifier
    for j in range(len(classifiers)):
        if i != j:
            S = stats.ttest_rel(vertical_result[i], vertical_result[j])
            p_values.append(S[1])

    with open('a1_3.4.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for result in fold_test_result_list:
            spamwriter.writerow(result)
        spamwriter.writerow(p_values)

        spamwriter.writerow(["The accuracy of the cross-validation's result may lead different result as part3.1 "+
                             "It could be caused by the variance of the data. In the 3.1, there are only one set of training"
                             " and testing data. The form of the trianing set may lead to bias."])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    ##### part1
    X_train, X_test, y_train, y_test, iBest = class31(args.input)

    ##### part2
    sizes = [1000, 5000, 10000, 15000, 20000]

    # store 1k train data
    train_1ks = []

    for size in sizes:
        train_1ks.append(class32(X_train[0:size], X_test[0:size], y_train, y_test, iBest))

    with open('a1_3.2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(part2_accuracy_list_1000292033)
        spamwriter.writerow(["The accuracy increases with the test size increate. However, " +
                             "It decreases a bit as the test size reach certain level." +
                             "It could be caused by overfiting."])

    ##### part3

    ks = [5, 10, 20, 30, 40, 50]

    for train_1k in train_1ks:
        class33(X_train, X_test, y_train, y_test, iBest, train_1k[0], train_1k[1])

    with open('a1_3.3.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in part3_feature_list_1000292033:
            spamwriter.writerow(row)
        spamwriter.writerow(part3_accuracy_list_1000292033)
        spamwriter.writerow(["The LIWC are chose in both situation. " +
                             "It may caused by that the LIWC are more close to the content of the words."])
        spamwriter.writerow(["p values are Lower with more data. With more data, we are more sure about the importance"
                             " of the features we have."])
        spamwriter.writerow(["p values are Lower with more data. With more data, we are more sure about the importance"
                             " of the features we have."])

    ##### part4

    class34(args.input, iBest)
