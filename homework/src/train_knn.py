#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from homework.src._internals.save_model import save_model

from ._internals.calculate_metrics import calculate_metrics

from ._internals.print_metrics import print_metrics
from ._internals.prepare_data import prepare_data

x_train, x_test, y_train, y_test = prepare_data()

# entrenar el modelo
estimator = KNeighborsRegressor(n_neighbors=5)
estimator.fit(x_train, y_train)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
y_pred = estimator.predict(x_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
save_model(estimator)

print()
print(estimator, ":", sep="")

mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
print_metrics(mse, mae, r2, "Metricas de entrenamiento:")

mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
print_metrics(mse, mae, r2, "Metricas de testing:")
