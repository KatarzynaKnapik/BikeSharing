import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LinearRegression
import math, datetime
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('day.csv')
df = df.set_index(['dteday'])

df = df[['season','mnth', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp','hum', 'windspeed','registered','cnt']]
forecast = 'cnt'

df.fillna(-99999, inplace=True)

nr_to_pred = int(math.ceil(len(df) * 0.1))

x = np.array(df.drop([forecast],1))
x = preprocessing.scale(x)
X = x[:-nr_to_pred]
X_rest = x[-nr_to_pred:]
y = np.array(df[forecast])[:-nr_to_pred]


# acc = 0
# while acc <= 0.98:
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15)
#     model = LinearRegression(n_jobs=-1)
#     model.fit(x_train, y_train)
#
#     acc = model.score(x_test, y_test)
#
# print(acc)
#
# with open('regressionmodel.pickle', 'wb') as f:
#         pickle.dump(model, f)

pickle_in = open('regressionmodel.pickle', 'rb')
model = pickle.load(pickle_in)
forecasted = model.predict(X_rest)

for i in range(len(forecasted)):
    print (y[-nr_to_pred:][i], forecasted[i])

last_date = df.iloc[-nr_to_pred].name
last_unix = datetime.datetime.strptime(last_date, "%Y-%m-%d").timestamp()
one_day = 86400
next_unix = last_unix + one_day


new_index = df['cnt'].tail(len(forecasted)).index
new_series = pd.DataFrame(index = new_index, data = forecasted)
new_series.columns=['forecast']
df = pd.merge(df, new_series, how = 'left', left_index=True, right_index=True)


df['cnt'].plot()
df['forecast'].plot()
plt.xlabel('Date')
plt.ylabel('No. of rental bikes including both casual and registered users')
plt.show()

