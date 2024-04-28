import pandas as pd


def processDS(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates()
    data.dropna(axis=0, how='all')
    data.dropna(axis=1, how='all')
    data.dropna(axis=0, how='any')
    data.dropna(axis=1, how='any')
    data = removeOutlier(data, "BMI")
    data.dropna(axis=0, how='any')
    return data

def removeOutlier(data, field):
    q1 = data[field].quantile(0.25)
    q3 = data[field].quantile(0.75)
    range = q3 - q1
    min = q1 - 1.5*range
    max = q3 + 1.5*range
    data = data.loc[(data[field] > min) & (data[field] < max)]
    return data

