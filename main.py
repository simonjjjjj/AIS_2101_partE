import exploring_dataset
import preprocess
import unsupervised_learning
import supervised_learning

if __name__ == '__main__':
    data = preprocess.processDS("diabetes_binary_health_indicators_BRFSS2015.csv")

    #exploring_dataset.analyzeData(data)

    #unsupervised_learning.kMeans(data)
    #unsupervised_learning.doDBSCAN(data)


    #exploring_dataset.hexbinPlot(data, 'BMI', 'Income')

    #supervised_learning.logisticRegression(data)




