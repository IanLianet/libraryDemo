import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns

df6 = pd.DataFrame(pd.read_csv('/Users/ianlai/Desktop/library/data6.csv', header=0, encoding='utf-8', dtype=str,
                               error_bad_lines=False))

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

print("\n**6***************df6.info*************")
df6.info()
print("\n***6**************df6.head*************")
print(df6.head())
print("\n****6*************df6.shape*************")
print(df6.shape)
print(df6.shape[0])
n_readers6 = df6.READER_ID.unique().shape[0]
n_books6 = df6.M_ISBN.unique().shape[0]
print("唯一用户数量：" + str(n_readers6) + " | 唯一图书数量：" + str(n_books6))

train_data, test_data = ms.train_test_split(df6, test_size=0.3)
train_data_matrix = np.zeros((n_readers6, n_books6), dtype=np.int)

for line in train_data.itertuples():
    train_data_matrix[int(line[21]) - 1, int(line[20]) - 1] = train_data_matrix[
                                                                  int(line[21]) - 1, int(line[20]) - 1] + 1

print('train_data_matrix***********\n%s' % train_data_matrix)
test_data_matrix = np.zeros((n_readers6, n_books6), dtype=np.int)
for line in test_data.itertuples():
    test_data_matrix[int(line[21]) - 1, int(line[20]) - 1] = test_data_matrix[int(line[21]) - 1, int(line[20]) - 1] + 1

print('test_data_matrix***********\n%s' % test_data_matrix)
# print(test_data_matrix[0, 25824])
print(np.nonzero(test_data_matrix))

# 计算相似度
# 使用sklearn的pairwise_distances函数来计算余弦相似性

print(train_data_matrix.shape)
item_similarity = pairwise_distances(train_data_matrix.T, metric='correlation')
user_similarity = pairwise_distances(train_data_matrix, metric='correlation')

print("item_similarity")
print(item_similarity)
print("user_similarity")
print(user_similarity)

item_row = item_similarity.shape[0]
item_col = item_similarity.shape[1]
user_row = user_similarity.shape[0]
user_col = user_similarity.shape[1]

for i in range(0, item_row):
    for j in range(0, item_col):
        if i == j | np.isnan(item_similarity[i, j]):
            item_similarity[i, j] = 0
        else:
            item_similarity[i, j] = 2 - item_similarity[i, j]

for i in range(0, user_row):
    for j in range(0, user_col):
        if i == j | np.isnan(item_similarity[i, j]):
            user_similarity[i, j] = 0
        else:
            user_similarity[i, j] = 2 - user_similarity[i, j]

print("item_similarity")
print(item_similarity)
print("user_similarity")
print(user_similarity)

for i in range(0, item_row):
    ind = np.argpartition(item_similarity[i, :], -5)[-5:]
    for j in range(0, item_col):
        if j not in ind:
            item_similarity[i, j] = 0

for i in range(0, user_row):
    ind = np.argpartition(user_similarity[i, :], -5)[-5:]
    for j in range(0, user_col):
        if j not in ind:
            user_similarity[i, j] = 0

print(np.nonzero(item_similarity[0, :]))
print(np.nonzero(item_similarity[1, :]))
print(np.nonzero(user_similarity[0, :]))
print(np.nonzero(user_similarity[1, :]))

print("***** user_simi")
print(user_similarity)
print("***** item_simi")
print(item_similarity)


def predict(ratings, similarity, type='user'):
    # 基于用户相似度矩阵的
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / (np.array(
            [np.abs(similarity).sum(axis=1)]) + 0.0001).T
    # 基于物品相似度矩阵
    elif type == 'item':
        pred = ratings.dot(similarity) / (np.array([np.abs(similarity).sum(axis=1)]) + 0.0001)  # 求评价均值
    return pred


print('***********结束计算similarity*******')
# 预测结果
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
print('/n*************item_prediction********')
print(item_prediction[0:8])


# print(user_prediction)


# 评估指标，均方根误差
# 使用sklearn的mean_square_error (MSE)函数，其中，RMSE仅仅是MSE的平方根
# 只是想要考虑测试数据集中的预测评分，因此，使用prediction[ground_truth.nonzero()]筛选出预测矩阵中的所有其他元素
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()  # 取测试矩阵相同的部分计算RMSE
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print('/n*****User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('/n*****Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
