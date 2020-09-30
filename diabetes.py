# Mathematics Plotting Library (?)
# Graficación (pyplot)
import matplotlib.pyplot as plt

# Manejador de datos, números y arreglos complejos
# Básicamente se maneja como Excel
import numpy as np

# Scikit-learn (sklearn) para aprendizaje de Machine Learning
# datasets -> conjuntos de datos de prueba
# linear_model -> algoritmos lineales (ej. LinearRegression)
from sklearn import datasets, linear_model
# mean_squared_error, r2_score -> evaluar precisión del algoritmo
from sklearn.metrics import mean_squared_error, r2_score


# Cargar datos de prueba de diabetes
# return_X_y=True -> variables separadas
# diabetes_X -> features (10) atributo + valor
# diabetes_y -> float que indica el progreso de diabetes (label con respuesta)
# Más información: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Reducir dimensionalidad de X a solo un feature
# 10 features ----> 1 feature
# La 3er columna (el índice 2)
# np.newaxis -> Incrementar la dimensionalidad en 1     
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Separar los datos en 2:
# 1-> Datos de entrenamiento (para que el algoritmo aprenda)
# 2-> Datos de prueba (evaluar al algoritmo)
# Puede ser un 80% - 20%


# Separar datos de entrenamiento
# En este caso serán 20 elementos de entrenamiento
# Del inicio hasta el (final - 20)
diabetes_X_train = diabetes_X[:-20]
# Del (final - 20) hasta el final
diabetes_X_test = diabetes_X[-20:]


# Separar datos de prueba
# En este caso serán 20 elementos de prueba
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Deben coincidir los elementos de entrenamiento vs los de prueba


# Crear objeto de algoritmo
# regr (regresión)
regr = linear_model.LinearRegression()

# Entrenar el modelo (con los datos de entrenamiento)
# fit() -> entrenar; pide dos datos:
# 1- Features (información que sirve para predecir)
# 2- Labels (respuestas)
regr.fit(diabetes_X_train, diabetes_y_train)

# El algoritmo ya fue entrenado, así que ya puede predecir datos

# Crear una variable con las "respuestas" de su predicción
# pred => predicción
# predict() => predecir datos a partir de los datos con los que se ha entrenado
# Utilizar los datos de prueba (evaluación) para las predicciones
# No podemos usar los mismos datos de entrenamiento, porque si los memorizó
# va a "sacarse un 10" (atinarle a la perfección, memorizar, etc.)
diabetes_y_pred = regr.predict(diabetes_X_test)

# print(diabetes_y_test[0])
# print(diabetes_y_pred[0])

# Calcular error
error = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print(f"Mean squared error: {error}")

# Coeficiente de determinación (1 es score perfecto, perfecta predicción)
score = r2_score(diabetes_y_test, diabetes_y_pred)
print(f"Coefficient of determination: {score}")


# Mostrar visualmente los resultados
# (usando graficación)

# Scatter -> Colocar muchos valores x, y (puntos)
# (Dispersión)
# Valores de prueba en X, y
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')

# Valores de entrenamiento en X, y
plt.scatter(diabetes_X_train, diabetes_y_train, color='purple')

# Línea (de regresión lineal)
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.show()