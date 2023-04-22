import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.datasets import load_breast_cancer

toPrint = False
numTrials = 200

def run(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    if(toPrint):
        print("  predictions:   ", *model.predict(x_test), sep='')
        print("  actual labels: ", *y_test, sep='')
    finalScore = model.score(x_test, y_test)
    if(toPrint):
        print("  FINAL SCORE: %0.4f" % finalScore)
        grade(finalScore*100)
        print()
    return finalScore

def main():
    cancer_set = load_breast_cancer()
    data = cancer_set.data
    target = cancer_set.target
    data, target = shuffle(data, target)

    N = 500

    x_train = data[:N]
    y_train = target[:N]
    x_test = data[N:]
    y_test = target[N:]
    # print(*y_test, sep='')

    nearestCentroidAvg = 0
    neighborsAvg = 0
    neighbors2Avg = 0
    decisionTreeAvg = 0
    randomForestAvg = 0
    for i in range(numTrials):
        data, target = shuffle(data, target)
        x_train = data[:N]
        y_train = target[:N]
        x_test = data[N:]
        y_test = target[N:]
        nearestCentroidAvg += run(x_train, y_train, x_test, y_test, NearestCentroid()) / numTrials
        neighborsAvg += run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3)) / numTrials
        neighbors2Avg += run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=10)) / numTrials
        decisionTreeAvg += run(x_train, y_train, x_test, y_test, DecisionTreeClassifier()) /numTrials
        randomForestAvg += run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators = 5)) / numTrials
    
    print("Nearest Centroid Average Score:")
    print("%0.4f"% nearestCentroidAvg)

    print("3-Neighbors Classifier:")
    print("%0.4f"% neighborsAvg)

    print("10-Neighbors Classifier:")
    print("%0.4f"% neighbors2Avg)

    print("Decision Tree Classifier:")
    print("%0.4f"% decisionTreeAvg)

    print("Random Forest Classifier:")
    print("%0.4f"% randomForestAvg)

    print()
    
main()