
import seaborn as sns
import matplotlib.pyplot as plt


def analyzeData(data):
    print(type(data))
    print(data.shape)
    print(data.columns)
    print(data.info())
    print(data.describe())

def correlationHeatmap(data):
    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features')
    plt.show()

def boxPlot(data, field):
    plt.boxplot(data[field])
    plt.show()

def getOutliers(data, field):
    outliers = plt.boxplot(data["Age"])["fliers"][0].get_data()[1]
    print(outliers)

def histogram(data, field):
    bins = range(12, 99)
    plt.hist(data["BMI"], bins)
    plt.show()

def scatterPlot(data, field1, field2):
    plt.xlabel(field1)
    plt.ylabel(field2)
    plt.scatter(data[field1], data[field2], color='blue', alpha=0.5)
    plt.show()

def hexbinPlot(data, field1, field2):
    plt.xlabel(field1)
    plt.ylabel(field2)
    plt.hexbin(data[field1], data[field2], gridsize=(25, 10), cmap="plasma")
    plt.show()