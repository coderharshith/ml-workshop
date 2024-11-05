from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler





import matplotlib.pyplot as plt

print("testing")

california  =  fetch_california_housing()
data  = pd.DataFrame(california.data, columns=california.feature_names)
# data.shape
print(data.shape)
print(california.target)
col = data.columns
print(col)
data['price'] = california.target


X_train,X_test,Y_train,Y_test = train_test_split(data,data.price,test_size=0.2)


#normalize the data
scaler=  StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)


#train the model
model = LinearRegression()
model.fit(X_train,Y_train)

output  = model.predict(X_test)
result = mean_squared_error(output,Y_test)
print(result)

#MLP
model  = Sequential()




# plt.scatter(X_test[:,0],Y_test,label='original data')
# plt.plot(X_test[:,0],output,label="predicted data",c='r')
# plt.show()
