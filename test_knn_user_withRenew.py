
import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

df = pd.DataFrame(pd.read_csv('/Users/ianlai/Desktop/library/data4.csv', header=0, encoding='utf-8', dtype=str,
                              error_bad_lines=False))
df5 = pd.DataFrame(pd.read_csv('/Users/ianlai/Desktop/library/data5.csv', header=0, encoding='utf-8', dtype=str,
                               error_bad_lines=False))
df6 = pd.DataFrame(pd.read_csv('/Users/ianlai/Desktop/library/data6.csv', header=0, encoding='utf-8', dtype=str,
                               error_bad_lines=False))
df2 = pd.DataFrame(pd.read_csv('/Users/ianlai/Desktop/library/data4_copy2.csv', header=0, encoding='utf-8', dtype=str,
                               error_bad_lines=False))
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

print("\n**6***************df6.info*************")
df6.info()
print("\n***6**************df6.head*************")
print(df6.head())
print("\n****6*************df6.shape*************")
print(df6.shape)
n_readers6 = df6.READER_ID.unique().shape[0]
n_books6 = df6.M_ISBN.unique().shape[0]
print("唯一用户数量：" + str(n_readers6) + " | 唯一图书数量：" + str(n_books6))

train_data, test_data = ms.train_test_split(df6, test_size=0.3)
train_data_matrix = np.zeros((n_readers6, n_books6), dtype=np.int)

for line in train_data.itertuples():
    train_data_matrix[int(line[21]) - 1, int(line[20]) - 1] = train_data_matrix[
                                                                  int(line[21]) - 1, int(line[20]) - 1] + 1 + int(line[8])

print('train_data_matrix***********\n%s' % train_data_matrix)
test_data_matrix = np.zeros((n_readers6, n_books6), dtype=np.int)
for line in test_data.itertuples():
    test_data_matrix[int(line[21]) - 1, int(line[20]) - 1] = test_data_matrix[int(line[21]) - 1, int(line[20]) - 1] + 1 + int(line[8])

print('test_data_matrix***********\n%s' % test_data_matrix)
# print(test_data_matrix[0, 25824])
print(np.nonzero(test_data_matrix))

# 计算相似度
# 使用sklearn的pairwise_distances函数来计算余弦相似性

print(train_data_matrix.shape)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(train_data_matrix)
distances, indices = nbrs.kneighbors(train_data_matrix)
print("**************indices******")
print(indices)
print("**************distances******")
print(distances)



