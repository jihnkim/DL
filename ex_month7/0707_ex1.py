# import pandas as pd
#
# df = pd.read_csv('high_diamond_ranked_10min.csv', sep=',')
#
# print(df.head())

"""
miniBatchSize?
Batch를 쪼갬
8개의 학습데이터가 있다면
4개를 한 덩어리(미니배치)에 넣고싶다?
그럼 두덩어리가 나옴

train데이터가 4개인데 minibatchsize가 4이면
1덩어리가나옴

"""
import matplotlib.pyplot as plt
import numpy as np

x = np.asarray([580, 700, 810, 840])
y_label_total = np.asarray([374, 385, 375, 401])

RND_STD = 1
RND_MEAN = 0

input_cnt = 1
output_cnt = 1

def main_execute(x, y, epoch_count, report, lr = 0.001):
    model_init() # theta value를 random 하게 추출함
    sse_row, theta_0_row, theta_1_row = run_train(x, y, epoch_count, report, lr)

    return sse_row, theta_0_row, theta_1_row

def model_init():
    global theta_0, theta_1
    theta_0 = np.random.normal(RND_MEAN, RND_STD, [output_cnt])
    theta_1 = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt]) # 행렬곱을 해야해서 차원 맞춰줌

def run_train(x, y, epoch_count, report, lr):
    print(f" theta_0 INITIALIZED : {theta_0}")
    print(f" theta_1 INITIALIZED : {theta_1}")

    sse_row = []
    theta_0_row = []
    theta_1_row = []

    for epoch in range(epoch_count):
        y_hat = forward_nn(x)
        sse = forward_postprocess(y_hat, y_label_total)

        sse_row.append(sse)

        back_propagation(y_hat, lr)

        theta_0_row.append(theta_0)
        theta_1_row.append(theta_1)

        if report > 0 and epoch % report == 0:
            print(f'EPOCH - {epoch + 1}')
            print(f'SSE : {sse}')

    print('===================================')
    print(f'FINAL SSE : {sse}')

    return sse_row, theta_0_row, theta_1_row

def forward_nn(x):
    y_hat = theta_0 + theta_1 * x

    return y_hat

def forward_postprocess(output, y): # 예측값과 실제 y 값 필요
    diff = output - y

    square = np.square(diff)

    sse = 1/2 * (np.sum(square))

    return sse

def back_propagation(y_hat, lr):
    global theta_0, theta_1
    theta_0 = theta_0 - lr * (np.sum(y_hat - y_label_total))
    theta_1 = theta_1 - lr * (np.sum(y_hat - y_label_total) * x)

sse_row, theta_0_row, theta_1_row = main_execute(x, y_label_total, epoch_count=10, report=2, lr=0.001)

plt.plot(sse_row, '--o', color = 'lightblue')
plt.xlabel('Epoch')
plt.ylabel('SSE')

plt.grid()

plt.show()