import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mtp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Función para entrenar el modelo y predecir la potencia activa
def predecir_potencia_activa():
    # Leer el archivo CSV con los datos de v_rms y i_rms
    datos_v = pd.read_csv("v_rms.csv")
    datos_i = pd.read_csv("i_rms.csv")
    datos_potencia = pd.read_csv("potencia_y_eficiencia.csv")

    # Combina los datos en un solo DataFrame
    datos = pd.concat([datos_v, datos_i, datos_potencia['potencia_activa']], axis=1)

    # Visualización de los datos
    sb.scatterplot(x="v_rms", y="potencia_activa", data=datos)
    sb.scatterplot(x="i_rms", y="potencia_activa", data=datos)
    mtp.show()

    # Preparar los datos para la regresión lineal
    X = datos[['v_rms', 'i_rms']]
    y = datos['potencia_activa']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Crear y entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Realizar predicciones y calcular el error
    y_pred = modelo.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    print(f'Error cuadrático medio: {error}')

    # Predicción de ejemplo
    v_rms_ejemplo = 250
    i_rms_ejemplo = 10
    potencia_predicha = modelo.predict([[v_rms_ejemplo, i_rms_ejemplo]])
    print(f"Potencia activa estimada: {potencia_predicha[0]}")

# Función para estimar la eficiencia de los paneles
def estimar_eficiencia():
    # Leer el archivo CSV con los datos de potencia activa y eficiencia
    datos = pd.read_csv("potencia_y_eficiencia.csv")

    # Visualización de los datos
    sb.scatterplot(x="potencia_activa", y="eficiencia", data=datos)
    mtp.show()

    # Preparar los datos para la regresión lineal
    X = datos[['potencia_activa']]
    y = datos['eficiencia']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Crear y entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Realizar predicciones y calcular el error
    y_pred = modelo.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    print(f'Error cuadrático medio: {error}')

    # Predicción de ejemplo
    potencia_activa_ejemplo = 1000
    eficiencia_predicha = modelo.predict([[potencia_activa_ejemplo]])
    print(f"Eficiencia estimada: {eficiencia_predicha[0]}")

# Función principal para ejecutar el modelo deseado
def main():
    print("Seleccione una opción:")
    print("1. Predecir potencia activa")
    print("2. Estimar eficiencia de los paneles")

    opcion = input("Ingrese el número de la opción seleccionada: ")

    if opcion == "1":
        predecir_potencia_activa()
    elif opcion == "2":
        estimar_eficiencia()
    else:
        print("Opción inválida. Por favor, seleccione una opción válida.")

if __name__ == "__main__":
    main()
