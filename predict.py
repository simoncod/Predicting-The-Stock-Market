import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("sphist.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'], ascending=True)
# print(df.head())

# create 3 indicators and set them to 0
df['day_5'] = 0
df['day_30'] = 0
df['day_365'] = 0

# Perform rolling function
df['day_5'] = df['Close'].rolling(5).mean()
df['day_5'] = df['day_5'].shift()
df['day_30'] = df['Close'].rolling(30).mean()
df['day_30'] = df['day_30'].shift()
df['day_365'] = df['Close'].rolling(365).mean()
df['day_365'] = df['day_365'].shift()
# print(df)

# Now drop any rows that contain na values
df = df.dropna(axis=0)
# check to see how many rows had been dropped
# print(df)

# Remove any rows from df that fall before 1951-01-03
new_df = df[df['Date'] > datetime(year=1951, month=1, day=2)]

# assign train and test. Set everything before 2013-01-01 to train and the rest to test
train_df = new_df[new_df['Date'] < datetime(year=2013, month=1, day=1)]
test_df = new_df[new_df['Date'] >= datetime(year=2013, month=1, day=1)]

# print(train_df.head(1))
# print(test_df.head(1))

# Now perform LinearRegression to calculate the mean square error from each of the prediction with different 'day' columns.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# single varible prediction on day_5
model = LinearRegression()
model.fit(train_df[['day_5']], train_df['Close'])
prediction = model.predict(test_df[['day_5']])
mse = mean_squared_error(prediction, test_df['Close'])
print(mse)

# single varible prediction on day_30
model = LinearRegression()
model.fit(train_df[['day_30']], train_df['Close'])
prediction = model.predict(test_df[['day_30']])
mse = mean_squared_error(test_df['Close'], prediction)
print(mse)

# single varible prediction on day_365
model = LinearRegression()
model.fit(train_df[['day_365']], train_df['Close'])
prediction = model.predict(test_df[['day_365']])
mse = mean_squared_error(test_df['Close'], prediction)
print(mse)

# finally let's calculate muiltivarible mse with these previous single varibles
model = LinearRegression()
model.fit(train_df[['day_5', 'day_30']], train_df['Close'])
prediction = model.predict(test_df[['day_5', 'day_30']],)
mse = mean_squared_error(test_df['Close'], prediction)
print(mse)

model = LinearRegression()
model.fit(train_df[['day_5', 'day_30', 'day_365']], train_df['Close'])
prediction = model.predict(test_df[['day_5', 'day_30', 'day_365']],)
mse = mean_squared_error(test_df['Close'], prediction)
print(mse)

