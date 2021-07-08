"""
Parameter - w / b
hyperparameter - lr . . .

deep learning process
1. data feature check
2. initialize parameters (weight, bias setting)
3. data split
4. minibatch process
5. forward propagation
5 ---> 2 back propagation
"""

import numpy as np
import pandas as pd

# data import
df = pd.read_csv('faults_mini.csv')
df.info()

data = np.asarray(df, dtype='float32')

print(type(data))
print(data)

# parameters setting
RND_MEAN = 0
RND_STD = 1

input_cnt = 4
output_cnt = 3

weight = np.random.normal(RND_MEAN, RND_STD, size = [input_cnt, output_cnt])
bias   = np.random.normal(RND_MEAN, RND_STD, size = [output_cnt])

print("weight.shape : ", weight.shape)
print("bais.shape : ", bias.shape)
print("data.shape : ", data.shape)

# mini batch
mb_size = 2
train_ratio = 0.8

MiniBath_step_count = int(data.shape[0] * train_ratio) // mb_size
print("MiniBath_step_count :", MiniBath_step_count)

test_begin_index = MiniBath_step_count * mb_size
print("test_begin_index : ",test_begin_index)

shuffle_map = np.arange(data.shape[0])
print("Before : shuffle_map : ",shuffle_map)

np.random.shuffle(shuffle_map)
print("After : shuffle_map : ", shuffle_map)

mb_data_1 = data[shuffle_map[0:2]]
mb_data_2 = data[shuffle_map[2:4]]
mb_data_3 = data[shuffle_map[4:6]]
mb_data_4 = data[shuffle_map[6:8]]

print("mb_data_1 \n", mb_data_1)
print("mb_data_2 \n", mb_data_2)
print("mb_data_3 \n", mb_data_3)
print("mb_data_4 \n", mb_data_4)

print("첫 번째 미니배치 데이터의 행과 열")
mb_1_train_x = mb_data_1[:, : -output_cnt]
mb_1_train_y = mb_data_1[:, -output_cnt : ]
print("mb_1_train_x : \n", mb_1_train_x)
print("mb_1_train_y : \n", mb_1_train_y)

print("두 번째 미니배치 데이터의 행과 열")
mb_2_train_x = mb_data_2[:, : -output_cnt]
mb_2_train_y = mb_data_2[:, -output_cnt : ]
print("mb_2_train_x : \n", mb_2_train_x)
print("mb_2_train_y : \n", mb_2_train_y)

print("세 번째 미니배치 데이터의 행과 열")
mb_3_train_x = mb_data_3[:, : -output_cnt]
mb_3_train_y = mb_data_3[:, -output_cnt : ]
print("mb_3_train_x : \n", mb_3_train_x)
print("mb_3_train_y : \n", mb_3_train_y)

print("네 번째 미니배치 데이터의 행과 열")
mb_4_train_x = mb_data_4[:, : -output_cnt]
mb_4_train_y = mb_data_4[:, -output_cnt : ]
print("mb_4_train_x : \n", mb_4_train_x)
print("mb_4_train_y : \n", mb_4_train_y)

# forward propagation
print("============첫 번째 미니배치 신경망 연산 결과(P1, P2, P3)=================")
mb_1_y_hat_1 = np.matmul(mb_1_train_x, weight[:,0]) + bias[0]
mb_1_y_hat_2 = np.matmul(mb_1_train_x, weight[:,1]) + bias[1]
mb_1_y_hat_3 = np.matmul(mb_1_train_x, weight[:,2]) + bias[2]

print("mb_1_y_hat_1 : ", mb_1_y_hat_1)
print("mb_1_y_hat_2 : ", mb_1_y_hat_2)
print("mb_1_y_hat_3 : ", mb_1_y_hat_3)

print("============두 번째 미니배치 신경망 연산 결과(P1, P2, P3)=================")
mb_2_y_hat_1 = np.matmul(mb_2_train_x, weight[:,0]) + bias[0]
mb_2_y_hat_2 = np.matmul(mb_2_train_x, weight[:,1]) + bias[1]
mb_2_y_hat_3 = np.matmul(mb_2_train_x, weight[:,2]) + bias[2]

print("mb_2_y_hat_1 : ", mb_2_y_hat_1)
print("mb_2_y_hat_2 : ", mb_2_y_hat_2)
print("mb_2_y_hat_3 : ", mb_2_y_hat_3)

print("============세 번째 미니배치 신경망 연산 결과(P1, P2, P3)=================")
mb_3_y_hat_1 = np.matmul(mb_3_train_x, weight[:,0]) + bias[0]
mb_3_y_hat_2 = np.matmul(mb_3_train_x, weight[:,1]) + bias[1]
mb_3_y_hat_3 = np.matmul(mb_3_train_x, weight[:,2]) + bias[2]

print("mb_3_y_hat_1 : ", mb_3_y_hat_1)
print("mb_3_y_hat_2 : ", mb_3_y_hat_2)
print("mb_3_y_hat_3 : ", mb_3_y_hat_3)

print("============네 번째 미니배치 신경망 연산 결과(P1, P2, P3)=================")
mb_4_y_hat_1 = np.matmul(mb_4_train_x, weight[:,0]) + bias[0]
mb_4_y_hat_2 = np.matmul(mb_4_train_x, weight[:,1]) + bias[1]
mb_4_y_hat_3 = np.matmul(mb_4_train_x, weight[:,2]) + bias[2]

print("mb_4_y_hat_1 : ", mb_4_y_hat_1)
print("mb_4_y_hat_2 : ", mb_4_y_hat_2)
print("mb_4_y_hat_3 : ", mb_4_y_hat_3)

#  -------------------------------------------------------------------------------------------------

mb_data_total = np.vstack((mb_data_1, mb_data_2, mb_data_3, mb_data_4)) # data migrate
mb_data_total_x = mb_data_total[:, :-3] # x, y split

print("mb_data_total_x.shape : ", mb_data_total_x.shape)

mb_total_y_hat = np.matmul(mb_data_total_x, weight) + bias
print(pd.DataFrame(mb_data_total_x))
print(mb_total_y_hat)

print(mb_1_y_hat_1, mb_1_y_hat_2, mb_1_y_hat_3)
print(mb_2_y_hat_1, mb_2_y_hat_2, mb_2_y_hat_3)
print(mb_3_y_hat_1, mb_3_y_hat_2, mb_3_y_hat_3)
print(mb_4_y_hat_1, mb_4_y_hat_2, mb_4_y_hat_3)