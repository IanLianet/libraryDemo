import numpy as np
import pandas as pd
import pymysql
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

df6 = pd.DataFrame(pd.read_csv('/RESOURCE_DIR/data6.csv', header=0, encoding='utf-8', dtype=str,
                               error_bad_lines=False))
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
# 获取唯一读者数量和唯一书籍数量，以便后面建立起读者-书籍矩阵
n_readers6 = df6.READER_ID.unique().shape[0]
n_books6 = df6.M_ISBN.unique().shape[0]
# 将数据分为训练集和测试集
train_data, test_data = ms.train_test_split(df6, test_size=0.3)
train_data_matrix = np.zeros((n_readers6, n_books6), dtype=np.int)
for line in train_data.itertuples():
    train_data_matrix[int(line[21]) - 1, int(line[20]) - 1] = train_data_matrix[
                                                                  int(line[21]) - 1, int(line[20]) - 1] + 1
test_data_matrix = np.zeros((n_readers6, n_books6), dtype=np.int)
for line in test_data.itertuples():
    test_data_matrix[int(line[21]) - 1, int(line[20]) - 1] = test_data_matrix[int(line[21]) - 1, int(line[20]) - 1] + 1
# 使用sklearn的pairwise_distances函数来计算欧几里得距离
item_distance = pairwise_distances(train_data_matrix.T, metric='euclidean')
user_distance = pairwise_distances(train_data_matrix, metric='euclidean')
# 将读者间的欧氏距离和书籍间的欧氏距离转化为读者间的相似度和书籍间的相似度
item_distance_row = item_distance.shape[0]
item_distance_col = item_distance.shape[1]
user_distance_row = user_distance.shape[0]
user_distance_col = user_distance.shape[1]
item_similarity = np.zeros((item_distance_row, item_distance_col), dtype=np.int)
user_similarity = np.zeros((user_distance_row, user_distance_col), dtype=np.int)
for i in range(0, item_distance_row):
    for j in range(0, item_distance_col):
        if item_distance[i, j] == 0:
            item_similarity[i, j] = 0
        else:
            item_similarity[i, j] = 1 / item_distance[i, j]
for i in range(0, user_distance_row):
    for j in range(0, user_distance_col):
        if user_distance[i, j] == 0:
            user_similarity[i, j] = 0
        else:
            user_similarity[i, j] = 1 / user_distance[i, j]
# 对于每一个书籍和读者，只选择与其相似度最高的top-k项来作为推荐的依据，将其余置为0
for i in range(0, item_distance_row):
    ind = np.argpartition(item_similarity[i, :], -3)[-3:]
    for j in range(0, item_distance_col):
        if j not in ind:
            item_similarity[i, j] = 0
for i in range(0, user_distance_row):
    ind = np.argpartition(user_similarity[i, :], -3)[-3:]
    for j in range(0, user_distance_col):
        if j not in ind:
            user_similarity[i, j] = 0


# 使用相似度矩阵，预测读者对每个书籍的评分
def predict(ratings, similarity, type='user'):
    # 基于读者相似度矩阵的
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        rating_prediction = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / (np.array(
            [np.abs(similarity).sum(axis=1)])).T
    # 基于书籍相似度矩阵
    elif type == 'item':
        rating_prediction = ratings.dot(similarity) / (np.array([np.abs(similarity).sum(axis=1)]))  # 求评价均值
    return rating_prediction


# 预测结果
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# 使用sklearn的mean_square_error (MSE)函数，进行一次平分根处理即为RMSE（均方根误差），以RMSE作为评估指标
# 只是想要考虑测试数据集中的预测评分，因此，使用prediction[ground_truth.nonzero()]筛选出预测矩阵中的所有其他元素
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()  # 取测试矩阵相同的部分计算RMSE
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print('/n*****User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('/n*****Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

item_recommendation = np.zeros((item_prediction.shape[0], 5), dtype=np.str)
user_recommendation = np.zeros((user_prediction.shape[0], 5), dtype=np.str)

x = 0
for i in range(0, item_prediction.shape[0]):
    ind = np.argpartition(item_prediction[i, :], -5)[-5:]
    for j in range(0, item_prediction.shape[1]):
        if j in ind:
            item_recommendation[i, x] = j

x = 0
for i in range(0, user_prediction.shape[0]):
    ind = np.argpartition(user_prediction[i, :], -5)[-5:]
    for j in range(0, user_prediction.shape[1]):
        if j in ind:
            user_recommendation[i, x] = j



# 打开数据库连接
db = pymysql.connect("localhost", "root", "youpass", "xlibrary")
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# SQL 更新语句
sql = 'INSERT INTO `recommendation`(`sid`,`book_id1`,`book_id2`,`book_id3`,`book_id4`,`book_id5`) VALUE(%s,%s,%s,%s,%s)'
# 对每一个读者执行SQL语句，以将推荐的书籍写入数据库内
for i in range(0, user_recommendation.shape[0]):
    cursor.execute(sql, (i, user_recommendation[i, 0], user_recommendation[i, 1], user_recommendation[i, 2],
                         user_recommendation[i, 3], user_recommendation[i, 4]))
    db.commit()

# 关闭数据库连接
db.close()
