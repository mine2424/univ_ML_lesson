import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

# カルフォルニアのカリフォルニアの住宅価格データを読み込む
housing = fetch_california_housing(as_frame = True)

# 説明変数
x_array = housing.data
# 目的変数
y_array = housing.target

# データセットの説明
#print(housing.DESCR)

model = LinearRegression()

x_train, x_test, y_train, y_test, = train_test_split(x_array, y_array, test_size = 0.4)

model.fit(x_train, y_train)

print(model.coef_[0])
print(model.intercept_)

y_pred = model.predict(x_test)

#plt.scatter(x_array, y_array)
#plt.show()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("r2:", '{:.5f}'.format(r2))
print("rmse:", '{:.5f}'.format(rmse))
print("mae:", '{:.5f}'.format(mae))
