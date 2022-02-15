import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# read cvs file
data = pd.read_csv("WineQT.csv")

# let's preview
print(data.head())

print(data.dtypes)
# data features astype float as we want.

print(data.info())
# There is no nun-null object therefore we don't have to fill, we don't have to delete rows that's include nun-null.

# We should drop "id" column because of it' s unnecessary.
data.drop(columns=["Id"], axis=1, inplace=True)

# Split quality values and drop it from dataframe
y = data.quality
data.drop(columns=["quality"], axis=1, inplace=True)

# let's check features scale
for column in data.columns:
    print("{}'s min-max value = ".format(column), min(data[column]), "-", max(data[column]))

# it' s not scaled 0-1
# We need to scale features  0 - 1 for optimization.
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
data = pd.DataFrame(scaled, columns=data.columns)

# Now we should check the value counts in the target variable.
print(y.value_counts(normalize=True))

#  Data is highly unbalanced and if we train on this data, our model will be highly biased.
#  We will utilize oversampling to overcome this challenge.
over_sample = SMOTE()
x, y = over_sample.fit_resample(data, y)

# Now our target is balanced
print(x.shape, y.shape)
print(y.value_counts())

# train_test_split for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# training on different classifier
score = []


def run_classifier(model):
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    print("{} Accuracy score".format(model), accuracy_score(y_test, predict))
    return score.append(accuracy_score(y_test, predict))


models = [KNeighborsClassifier(), SVC(), RandomForestClassifier(), DecisionTreeClassifier(), GaussianNB()]

for model in models:
    model_ = model
    run_classifier(model_)

print("best model=", models[score.index(max(score))], "best score=", max(score))
