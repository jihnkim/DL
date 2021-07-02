# 행렬곱을 수행하는 대표 함수
# np.matmul
# np.dot

# 추정치 y >>> pred_y, y_hat, target_y
# 실제값 y >>> label_y

import csv
import numpy as np

with open('abalone_mini.csv') as csvfile:
  csvreader = csv.reader(csvfile)

  rows = []

  for row in csvreader:
      rows.append(row)




data = np.zeros([5, 11])

for n, row in enumerate(rows):
    if row[0] == 'M':
        data[n, 0] = 1
    elif row[0] == 'F':
        data[n, 1] = 1
    else:
        data[n, 2] = 1

    data[n, 3:] = row[1:]

print(data)
# ----------------------------------------------------------------------------

RND_MEAN = 0
RND_STD = 1

input_x = 10
output_y = 1

# data import
def import_data():
    pass

# initialize model weight, bias
def model_init():
    global weight, bias
    weight = np.random.normal(RND_MEAN, RND_STD, size=[input_x, output_y])
    bias = np.random.normal(RND_MEAN, RND_STD, size=[output_y])

# predict y
def forward_neuralnet(x):
    y_hat = np.matmul(x, weight) + bias
    return y_hat

# loss
def forward_postproc(output, y):
    print('output : \n', output)
    print('y : \n', y)

    diff = output - y
    print('diff : \n', diff)

    square = np.square(diff)
    print('square : \n', square)

    mse = np.mean(square)
    print('mse : \n', mse)

    return mse

def run_train(x, y):
    output = forward_neuralnet(x)
    loss = forward_postproc(output, y)

    return output, loss

def main_exec(x, y):
    import_data()
    model_init()
    run_train(x, y)

# 출력
main_exec(data[:, :-1], data[:, -1:])