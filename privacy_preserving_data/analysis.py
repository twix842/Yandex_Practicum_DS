import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy import linalg
from IPython.display import display

pd.set_option("display.precision", 3)
warnings.filterwarnings("ignore")

try:
    data = pd.read_csv("/Users/kolotukhin.md/Downloads/jupyter_notebook/8/insurance.csv")
except:
    data = pd.read_csv("https://code.s3.yandex.net/datasets/insurance.csv")

data.info()
data.head()

data = data.astype({"Возраст": "int64", "Зарплата": "int64"})
display(data.isnull().sum())
print(data.duplicated().value_counts())

features = data.drop(["Страховые выплаты"], axis=1)
target = data["Страховые выплаты"]

random_matrix = np.random.normal(size=(4, 4))
inv_random_matrix = linalg.inv(random_matrix)

encoded_features = np.dot(features, random_matrix)
decoded_features = np.dot(encoded_features, inv_random_matrix)

print("Original feature matrix")
display(pd.DataFrame(features))
print("------------------------")
print("Encoded feature matrix")
display(pd.DataFrame(encoded_features))
print("------------------------")
print("Decoded feature matrix")
display(pd.DataFrame(decoded_features))

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

features_train_encoded, features_test_encoded, target_train_encoded, target_test_encoded = train_test_split(
    encoded_features, target, test_size=0.25, random_state=12345
)

model = LinearRegression()

model.fit(features_train, target_train)
predictions = model.predict(features_test)
print("R2 on original data:", round(r2_score(target_test, predictions), 3))

model.fit(features_train_encoded, target_train_encoded)
predictions = model.predict(features_test_encoded)
print("R2 on transformed data:", round(r2_score(target_test_encoded, predictions), 3))
