from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from IPython.display import clear_output
from IPython.core.display import display, HTML

rank = 10
n_client = 40
alpha = 1000
label_distribution = np.random.dirichlet([alpha]*rank, 1)
print(label_distribution)
plt.plot(label_distribution[0])
alpha = 100
label_distribution = np.random.dirichlet([alpha]*rank, 1)
print(label_distribution)
plt.plot(label_distribution[0])
alpha = 10
label_distribution = np.random.dirichlet([alpha]*rank, 1)
print(label_distribution)
plt.plot(label_distribution[0])
alpha = 1
label_distribution = np.random.dirichlet([alpha]*rank, 1)
print(label_distribution)
plt.plot(label_distribution[0])
alpha = 0.1
label_distribution = np.random.dirichlet([alpha]*rank, 1)
print(label_distribution)
plt.plot(label_distribution[0])

plt.show()