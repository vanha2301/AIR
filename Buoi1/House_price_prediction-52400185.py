import numpy as np
import matplotlib.pyplot as plt

# ---------- Chuẩn hóa ----------
def add_bias(X):
    # thêm cột 1 ở đầu cho bias (w0)
    return np.c_[np.ones((X.shape[0], 1)), X]

def standardize(X):
    # chuẩn hoá Z-score: (x - mean)/std
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0  # tránh chia 0
    return (X - mu) / sigma, mu, sigma

def loss_mse(Xb, y, w):
    # J(w) = 1/(2N) * ||Xb w - y||^2
    N = len(y)
    y_pred = Xb @ w
    return np.sum((y_pred - y) ** 2) / (2 * N)

def gradientDescent_multi(X, y, lr=0.01, epochs=2000):
    # chuẩn hoá feature để dễ hội tụ
    Xn, mu, sigma = standardize(X)
    Xb = add_bias(Xn)                   # (N, d+1)
    N, d1 = Xb.shape

    # khởi tạo w
    w = np.zeros(d1)
    losses = []

    for _ in range(epochs):
        y_pred = Xb @ w
        grad = (Xb.T @ (y_pred - y)) / N   # gradient
        w -= lr * grad
        losses.append(loss_mse(Xb, y, w))

    return w, losses, mu, sigma

"""
    Khai tác dữ liệu từ file csv
"""
import pandas as pd

df = pd.read_csv(r"D:\Scr-AIR\Buoi1\data.csv")

#lượt bỏ date và các thông tin có dạng text 
X = df.drop(['price', 'date', 'street', 'city', 'statezip', 'country'], axis=1).to_numpy()
print(X)
Y = df['price'].to_numpy()
print('Y=', Y)

# huấn luyện
w, losses, mu, sigma = gradientDescent_multi(X, Y, lr=0.01, epochs=2000)
print('w =', w)

# vẽ đồ thị hàm mất mát
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()


