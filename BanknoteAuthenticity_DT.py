import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


dataset = 'data_banknote_authentication.txt'

cols = ["Wavelet_Var", "Wavelet_Skew", "Wavelet_kurtosis", "IMG_Entropy", "label"]

dataset = pd.read_csv(dataset, delimiter=',', header=None, names=cols)

dataset.head()


features =  ["Wavelet_Var", "Wavelet_Skew", "Wavelet_kurtosis", "IMG_Entropy"]
X = dataset[features]
y = dataset.label


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

DTC = DecisionTreeClassifier()

DTC.fit(X_train, y_train)

preds = DTC.predict(X_test)

print("Acc:", accuracy_score(y_test, preds))