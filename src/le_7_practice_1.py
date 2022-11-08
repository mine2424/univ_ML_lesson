import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

x = [[30], [35], [45], [55], [60], [65], [75], [80], [85], [95], [100]]
y = [29.7, 40.6, 41.85, 65.25, 63.2, 65.65, 68.25, 101.6, 79.9, 119.6, 123]


model = LinearRegression()

x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size = 0.4)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.scatter(x, y)
plt.show()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("r2:", '{:.5f}'.format(r2))
print("rmse:", '{:.5f}'.format(rmse))
print("mae:", '{:.5f}'.format(mae))
