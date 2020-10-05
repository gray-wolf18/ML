import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style 
from sklearn import svm

style.use('ggplot')

my_input= np.array([[3,2],[6,6],[2.6,3],[7,8],[3.5,5],[6,11]])
my_output = [0,1,0,1,0,1]

my_model = svm.SVR(kernel='linear')
my_model.fit(my_input,my_output)

print(my_model.predict([[8.5,10]]))
print(my_model.predict([[0.5,0.8]]))

plt.scatter(my_input[:,0],my_input[:,1], c = my_output)
plt.scatter(0.5,0.8 ,c='r')
plt.scatter(8.5,10 ,c='r')
plt.show()