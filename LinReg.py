#Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle

#This reads the CSV file. I included the skip parameter since I was having an error reading the file
dataset = pd.read_csv('Car_Regression_Data.csv', on_bad_lines='skip')

#Generating heatmap
sns.heatmap(dataset[['year', 'sellingprice', 'odometer', 'condition', 'mmr']].corr())
plt.title('Correlation Matrix Heatmap')
plt.show()

#Generating scatterplot
dataset.plot(kind='scatter', x = 'condition', y = 'sellingprice')
plt.title('Car Condition vs Selling Price')
plt.show()

#Generating pie chart of transmissions
pieData = dataset.transmission.value_counts()
pieData.plot.pie(title='Pie Chart Showing the Distribution of Transmissions')
plt.ylabel("")
plt.show()

#Generating a histogram
dataset["condition"].hist(bins=30, edgecolor='black', figsize=(10, 6))
plt.xlabel("Condition")
plt.ylabel("Frequency")
plt.title("Condition Score Histogram")
plt.show()

#Dropping all the unneccessary columns
dataset = dataset.drop("color", axis = 1)
dataset = dataset.drop("interior", axis = 1)
dataset = dataset.drop("saledate", axis = 1)
dataset = dataset.drop("state", axis = 1)
dataset = dataset.drop("vin", axis = 1)
dataset = dataset.drop("trim", axis = 1)
dataset = dataset.drop("model", axis = 1)
dataset = dataset.drop("make", axis = 1)
dataset = dataset.drop("seller", axis = 1)

#Dropping NaNs
dataset = dataset.dropna()

#Encoding the transmission
transmissionEncoding = pd.get_dummies(dataset['transmission'], dtype='int')
#Dropping the original column
dataset = dataset.drop("transmission", axis = 1)
#Joining the encoded columns
dataset = dataset.join(transmissionEncoding)

#Encoding the body type
bodyEncoding = pd.get_dummies(dataset["body"], dtype = "int")
#Dropping the original column
dataset = dataset.drop("body", axis = 1)
#Joining the encoded columns
dataset = dataset.join(bodyEncoding)

#Separating the x and y values
xVals = dataset.drop("sellingprice", axis = 1)
yVals = dataset["sellingprice"]

#Creating the training and testing sets on a 70:30 split
xTrain, xTest, yTrain, yTest = train_test_split(xVals, yVals, test_size = 0.30, random_state=2)

#Creating the model
regModel = LinearRegression()

#Training the model
regModel.fit(xTrain, yTrain)

#Predictions on the test data
yPredicted = regModel.predict(xTest)

#Assesing various metrics
r2 = r2_score(yTest, yPredicted)
print(f'This model has an R^2 value of {r2}.')

mse = mean_squared_error(yTest, yPredicted)
print(f'This model has an MSE of {mse}.')

rmse = mean_squared_error(yTest, yPredicted, squared=False)
print(f'This model has an RMSE of {rmse}.')

mae = mean_absolute_error(yTest, yPredicted)
print(f'This model has an MAE of {mae}.')

#This exports the model as a pkl file to be deployed on the web
with open('car_price_prediction_model.pkl', 'wb') as f:
  pickle.dump(regModel, f)
