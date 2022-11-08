# 20221107 ML_2 lesson 7

# used all lib in lesson7
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

x = [[0], [1], [2], [3], [4], [5]]
y = [3, 6, 9, 12, 15, 18]

r = np.random.randint(-5,5,len(y))
y = y+r

# サンプルの描画
#plt.scatter(x,y)
#plt.show()

# 学習モデルを生成する
model = LinearRegression()

# 学習データをもとに最適なパラメータを求める
#model.fit(x, y)

# 求めたパラメータを出力する
#print("傾き:", model.coef_[0])
#print("切片:", model.intercept_)

# insert predict value
#y_pred = model.predict([[4]])
#print("predict value: ", y_pred)

# ４対１で学習データと評価データに分割する。
# train -> leaning data, test -> evaluateing data
# x -> x variable data, y -> y variable data
X_train, X_test, y_train, y_test, = train_test_split (x, y, test_size = 0.4)

# 結果の出力
#print("学習データ", X_train)
#print("評価データ", X_test)


# 学習データをもとに最適なパラメータを求める
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 学習データの描画
plt.scatter(X_train, y_train, color= 'blue')
# 予測結果の描画
plt.scatter(X_test, y_pred, color='red')
#グラフの表示
plt.show()

# 決定係数を求める
# ただの線形モデルなら使える
# predict value and evaluating value
r2 = r2_score(y_test, y_pred)

# MSE の算出(誤差からの計算/誤差が0なら0)
# 非線形の場合
mse = mean_squared_error(y_test, y_pred)

# RMSE の算出。 Numpy の sqrt 関数を使って、 MSE の平方根を取る
rmse = np.sqrt(mse)

# MAE の算出
# ただの差分なので、非線形で大きな幅の値の影響を受けないため
mae = mean_absolute_error(y_test, y_pred)

#print("r2:", '{:.5f}'.format(r2))
#print("rmse:", '{:.5f}'.format(rmse))
#print("mae:", '{:.5f}'.format(mae))


