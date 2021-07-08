import numpy as np
import pandas as pd

df = pd.read_csv('faults.csv')
df.info()

df_dataset = df[['X_Minimum','X_Maximum','Steel_Plate_Thickness','LogOfAreas',
                 'Pastry','Z_Scratch','Bumps']]
print("df_dataset.shape : ", df_dataset.shape)
print(df_dataset.info())

# 데이터 셋의 일부를 뽑아 독립, 종속변수 설정

df_dataset = np.asarray(df_dataset, dtype='float32')

df_dataset_x = df_dataset[0:1, :-3]
df_dataset_y = df_dataset[0:1, -3:]


print(f'독립변수 : {df_dataset_x}')
print(f'독립변수 shape : {df_dataset_x.shape}')
print(f'종속변수 : {df_dataset_y}')
print(f'종속변수 shape : {df_dataset_y.shape}')

# 초기 값 설정

RND_STD = 1
RND_MEAN = 0

input_cnt = df_dataset_x.shape[-1]
output_cnt = df_dataset_y.shape[-1]

weight = np.random.normal(RND_MEAN, RND_STD, size=[input_cnt, output_cnt]) # size는 입력 변수 개수, 출력 값 개수
bias = np.random.normal(RND_MEAN, RND_STD, size=[output_cnt])

print(f'weight.shape : {weight.shape}')
print(f'bias.shape : {bias.shape}')
print(f'df_dataset_x :\n {df_dataset_x}')
print(f'weight :\n {weight}')
print(f'bias :\n {bias}')

# 행렬 곱
print("df_dataset_x : \n", df_dataset_x) # 두개의 차이 >> 차원 수
print("="*20)
print("df_dataset_x[0] : \n", df_dataset_x[0])

p_1 = np.matmul(df_dataset_x[0], weight[:, 0]) + bias[0]
p_2 = np.matmul(df_dataset_x[0], weight[:, 1]) + bias[1]
p_3 = np.matmul(df_dataset_x[0], weight[:, 2]) + bias[2]

p_total = np.matmul(df_dataset_x, weight) + bias
print(p_total)
print(df_dataset_y)
