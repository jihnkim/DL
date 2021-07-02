import numpy as np

kor_y, kor_y_ = 15, 21
usa_y, usa_y_ = 12, 10
chn_y, chn_y_ = 18, 14

total_MSE = ((kor_y-kor_y_)**2 + (usa_y-usa_y_)**2 + (chn_y-chn_y_)**2) /3

print(np.round(total_MSE, 4))

# Vector > 한 줄
# scholar > 한 개
# Matrix > 한 행렬
# One - hot Vector > 선형성의 문제를 해결가능 하지만 희소 벡터, 메모리 비효율 문제가 발생
# ---------------------------------------------------------------
data_lst = ['M', 0.455, 0.365, 0.095]
new_lst =[]

data_arr = np.zeros(6,)

if data_lst[0] == 'M':
  data_arr[0] = 1
elif data_lst[0] == 'F':
  data_arr[1] = 1
else:
  data_arr[2] = 1

print(data_arr)

new_lst = list(data_arr)

new_lst[3:6] = data_lst[1:4]

print(new_lst)
# ---------------------------------------------------------------
