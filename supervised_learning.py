
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def logisticRegression(data):
    x_train, x_test, y_train, y_test = splitData(data)

    logistic_reg_model = LogisticRegression(penalty='l1', max_iter=250000, C=10, solver='saga', class_weight='balanced')
    logistic_reg_model.fit(x_train, y_train)

    y_pred = logistic_reg_model.predict(x_test)

    print("Prediction: ", y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def randomForest(data):
    x_train, x_test, y_train, y_test = splitData(data)

    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=30, max_depth=7, max_features=0.8, class_weight='balanced')
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)

    print("Prediction: ", y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def splitData(data):
    x = data.drop(columns=['Diabetes_binary'])
    y = data['Diabetes_binary']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("Size of training set:", len(x_train))
    print("Size of testing set:", len(x_test))
    return x_train, x_test, y_train, y_test