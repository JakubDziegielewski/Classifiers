from experiments.experiments import run_experiments

def main():
    list = [("glass.arff", [10, 10, 10, 10, 10, 10, 8, 10, 8]), ("wdbc.arff", [4]*30), ("twodiamonds.arff", [8]*2)]
    run_experiments(list, col_to_drop=[None, "IDNumber", None])

if __name__=="__main__":
    main()
