import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style 
from sklearn.decomposition import PCA

rng = np.random.RandomState(1)
style.use('ggplot')
my_input = np.dot(rng.rand(2,2),rng.randn(2,100)).T

my_model = PCA(n_components=1)
my_model.fit(my_input)

my_input_pca = my_model.transform(my_input)
print(my_input.shape)
print(my_input_pca.shape)

my_new = my_model.inverse_transform(my_input_pca)

#my_input_new = my_model.inverse_transform(my_input_pca)

plt.scatter(my_input[:,0],my_input[: , 1] )
plt.scatter(my_new[: , 0], my_new[: ,1])
plt.show()