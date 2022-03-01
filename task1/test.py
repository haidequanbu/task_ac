from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np

def model_liner1():
    X=np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y=np.dot(X,np.array([1,2]))+3
    reg=LinearRegression().fit(X,y)
    # reg.coef_
    # reg.intercept
    print(reg.predict(np.array([[3,5]])))

# 暂未完成的实验
# def model_liner2():
#     A=np.reshape(np.arange(100),[-1,1])
#     y=0.3*A+1
#     reg=LinearRegression().fit(A,y)
#     print(reg.predict(np.array([1])))
#
def model_svr():
    X=np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y=np.dot(X,np.array([1,2]))+3
    model = SVR(kernel='rbf',gamma=1,C=100)
    model.fit(X, y)
    print(model.predict([[3,4]]))



if __name__=="__main__":
    model_svr()
    # model_liner2()
    # print(sk.__version__)