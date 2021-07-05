import matplotlib.pyplot as plt
import numpy as np


class Y_pred():
    def __init__(self, theta_0, theta_1):
        self.theta_0 = theta_0
        self.theta_1 = theta_1


def out_y_hat(self, input_x):
    y_hat_row = []
    for i in range(len(input_x)):
        y_hat = self.theta_0 + self.theta_1 * input_x[i]
        y_hat_row.append(y_hat)

        return y_hat_row

Y_pred.y_hat = out_y_hat

Y_pred_C = Y_pred(theta_0=1,theta_1=2)

x = [1000]

print(Y_pred_C.y_hat(x))

"""
목적함수 >> 정략적 지표를 높일 것인지 낮출것인지에 대하여 목적을 가지고 있는 함수
손실함수 >> 신경망의 성능을 측정하고자 출력한 정량적 지표
"""

# SSE 구해보기

y_label_total = [374, 385, 375, 401]
y_hat_total = [1161, 1401, 1621, 1681]

diff_row = []

for i in range(len(y_label_total)):
    diff = y_label_total[i] - y_hat_total[i]
    diff_row.append(diff)
    square = np.square(diff_row)
    sse = 1/2*np.sum(square)

print(f'Diff : {diff_row}\n Square : {square}\n SSE : {sse}')

"""
Learning rate : 값의 갱신되는 보폭을 제어하는 역할
"""

def g(x, learning_rate):
    print(f'Now x value : {x}')
    epoch_1_x = x - learning_rate * ((2*x)-2)
    print(f'Epoch 1 -x : {epoch_1_x}')

    epoch_2_x = epoch_1_x - learning_rate * ((2 * epoch_1_x) - 2)
    print(f'Epoch 2 -x : {epoch_2_x}')

    epoch_3_x = epoch_2_x - learning_rate * ((2 * epoch_2_x) - 2)
    print(f'Epoch 3 -x : {epoch_3_x}')

    epoch_4_x = epoch_3_x - learning_rate * ((2 * epoch_3_x) - 2)
    print(f'Epoch 4 -x : {epoch_4_x}')


g(3, 0.1)

# 반복문 사용 def 만들기

def g1(learning_rate, epoch_count,random_value_bool = False, x=3):
    if random_value_bool:
        x = np.random.normal(0, 1, size=1)
    print(f'Now x value : {x}')
    print(f'Learning rate : {learning_rate}')
    x_row = []
    for i in range(epoch_count):
        x = x - learning_rate *((2*x)-2)
        print(f'Epoch : {i+1} / x : {np.round(x)}')
        x_row.append(x)

    return x_row

result = g1(0.1, 20, True)

print(result)

plt.plot(result, '--o', color='green')
plt.xlabel('Epoch')
plt.ylabel('X_value')
plt.grid()
plt.show()

# 매개변수 갱신식
