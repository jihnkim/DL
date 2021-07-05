import csv
import numpy as np

with open('C:\Users\sysan\OneDrive\바탕 화면\class\Deep & Reinforce Learning\dataset-20210630T001027Z-001\dataset\abalone_mini.csv') as csvfile:
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

