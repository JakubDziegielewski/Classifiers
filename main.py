from experiments.experiments import run_experiments

def main():
    list = [("glass.arff", [10, 10, 10, 10, 10, 10, 8, 10, 8, 10]), ("wdbc.arff", [4]*30)]
    run_experiments(list, col_to_drop=[None, "IDNumber"])

if __name__=="__main__":
    main()


"""
x_train = np.loadtxt("data/X_train.txt", dtype = float)
x_test = np.loadtxt("data/X_test.txt", dtype = float)
y_train = np.loadtxt("data/y_train.txt", dtype = int)
y_test = np.loadtxt("data/y_test.txt", dtype = int)
verbose = True
nb = NaiveBayes()
intervals = find_intervals(x_train, [4] * 561)
x_train = np.array(
    [
        data_discretization(features, intervals[i])
        for i, features in enumerate(x_train.T)
    ]
).T
x_test = np.array(
    [
        data_discretization(features, intervals[i])
        for i, features in enumerate(x_test.T)
    ]
).T
if verbose:
    print(f"Testing discrete naive bayes classifier")
    print("result of classification of the test set:")
good = 0
total = 0
nb = NaiveBayes()
nb.fit(x_train, y_train - 1)
predictions = nb.predict(x_test)

if verbose:
    print(f"Accuracy: {sum(predictions == y_test - 1) / len(predictions)}")
"""