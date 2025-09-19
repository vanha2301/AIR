import pandas as pd

df = pd.read_csv(r"D:\Scr-AIR\Buoi1\data.csv")

#lượt bỏ date và các thông tin có dạng text 
X = df.drop(['price', 'date', 'street', 'city', 'statezip', 'country'], axis=1).to_numpy()
print(X)
Y = df['price'].to_numpy()
Y = Y.reshape(-1, 1)

print('Y=', Y)
