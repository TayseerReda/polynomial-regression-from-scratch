
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


res=[]
L = []

def compination(indx,degree,x,ind):
    if(indx==degree):
        return

    for i in range(ind,len(x), 1):
          L.append(x[i])
          compination(indx+1,degree,x,i)
          #print(L)
          result = 1
          for num in L:
              result *= num
          res.append(result)

          L.pop()



class LinearRegression() :

    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


df=pd.read_csv("/content/SuperMarketSales.csv")
df.dropna(how='any',inplace=True)
df.drop_duplicates(inplace = True)

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.day

col="Weekly_Sales"
X=df.loc[:, df.columns != col]

corr = df.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Weekly_Sales'])>0.02]
#Correlation plot
#plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
#plt.show()
top_feature = top_feature.delete(-1)
print(top_feature)
X = X[top_feature]


sc = StandardScaler()
X = sc.fit_transform(X)
X=pd.DataFrame(X)

Y = df['Weekly_Sales']
#print(type(X))
Xcombined=[]
for i in range(len(X)):
    pas=X.iloc[i]
    pas=pas.values.tolist()
    #print(type(pas))
    compination(0,2,pas,0)
    Xcombined.append(res)
    res=[]
    #res.clear()



poly_features=pd.DataFrame(Xcombined)

X_train, X_test, y_train, y_test = train_test_split(poly_features, Y, test_size=0.2,random_state=0)
model = LinearRegression(0.0000002,100)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
print('Mean Square Error', metrics.mean_squared_error(y_test, Y_pred))



