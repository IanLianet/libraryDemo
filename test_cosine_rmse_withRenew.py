
import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
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

print("\n*****************df.info*************")

df.info()
print("\n*****************df.head*************")
print(df.head())
print("\n*****************df.shape*************")
print(df.shape)

# x = [16, 17]
# df.drop(df.columns[x], axis=1, inplace=True)
# print(df)

# df.dropna(axis=0, subset=[''])
print("\n*****************df.isnull().any()*************")
df = df.replace('null', np.NaN)
print(df.isnull().any())

# df.dropna(subset=['READER_ID'])


# num = df[[16]].notnull()
# df = df[df[[16]].notnull() & (df[[16]] != "")]
# df = df[df[[17]].notnull() & (df[[17]] != "")]
# df.info()

print("\n*****************var*************")
# df = df[df.iloc[:, [16, 17]].notnull()]
# print(df)
var = df[(df.iloc[[16]].notnull())].index.tolist()
# print(var)
# print(df.iloc[var, :])

print("\n**************唯一用户数量以及唯一图书数量****************")
n_readers = df.READER_ID.unique().shape[0]
n_books = df.M_ISBN.unique().shape[0]
print("唯一用户数量：" + str(n_readers) + " | 唯一图书数量：" + str(n_books))

print("\n**2222***************df2.info*************")
df2.info()
print("\n***2222**************df2.head*************")
print(df2.head())
print("\n****2222*************df2.shape*************")
print(df2.shape)
n_readers2 = df2.READER_ID.unique().shape[0]
n_books2 = df2.M_ISBN.unique().shape[0]
print("唯一用户数量：" + str(n_readers2) + " | 唯一图书数量：" + str(n_books2))

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
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')


def predict(ratings, similarity, type='user'):
    # 基于用户相似度矩阵的
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    # 基于物品相似度矩阵
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])  # 求评价均值
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
    prediction = prediction[ground_truth.nonzero()].flatten()#取测试矩阵相同的部分计算RMSE
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print('/n*****User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('/n*****Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

