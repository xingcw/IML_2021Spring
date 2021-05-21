import numpy as np
import pandas as pd


def main():
    path1 = 'prob_predictions.csv'
    path2 = 'proba13.csv'
    path3 = 'proba10.csv'
    df1 = pd.read_csv(path1, header=None).values
    df2 = pd.read_csv(path2, header=None).values
    df3 = pd.read_csv(path3, header=None).values
    merge = np.mean(np.asarray([df1, df2, df3]).squeeze(), axis=0)
    merge = np.where(merge >= 0.5, 1, 0)
    pd.DataFrame(merge).to_csv('merge_blosum_10_13.csv', index=False, header=None)


def compare():
    path1 = 'merge_13_10.csv'
    path2 = 'merge_blosum_10_13.csv'
    df1 = pd.read_csv(path1, header=None).values
    df2 = pd.read_csv(path2, header=None).values
    diff = np.sum(np.abs(df1 - df2))
    print(diff)


if __name__ == '__main__':
    compare()