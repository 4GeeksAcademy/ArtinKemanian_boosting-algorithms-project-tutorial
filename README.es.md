# Boosting - Guía paso a paso

- Utiliza los datos que hemos analizado en los dos proyectos anteriores.
- Continúa con el desarrollo para buscar un modelo que se adapte mejor.

## 🌱  Cómo iniciar este proyecto

Sigue las siguientes instrucciones:

1. Crea un nuevo repositorio basado en el [proyecto de Machine Learing](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Abre el repositorio creado recientemente en Codespace usando la [extensión del botón de Codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Una vez que el VSCode del Codespace haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver el caso práctico, asegúrate de confirmar tus cambios, haz push a tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instrucciones

### Prediciendo la diabetes

En los dos proyectos anteriores vimos cómo podíamos utilizar un árbol de decisión y después un random forest para mejorar la predicción de la diabetes. Hemos llegado a un punto en el que necesitamos mejorar. ¿Puede ser boosting la mejor de las alternativas para optimizar los resultados?

Boosting es una composición de modelos (generalmente árboles de decisión) secuencial en la cual el modelo nuevo persigue corregir los errores del anterior. Puede que esta visión nos sea útil en este conjunto de datos, ya que se cumplen varias de las suposiciones estudiadas en el módulo.

En este proyecto te centrarás en esta idea entrenando el conjunto de datos para mejorar el $accuracy$.

Recuerda que los proyectos anteriores puedes encontrarlos [aquí](https://github.com/4GeeksAcademy/decision-tree-project-tutorial) (árboles de decisión) y [aquí](https://github.com/4GeeksAcademy/random-forest-project-tutorial) (random forest).

#### Paso 1: Carga del conjunto de datos

Carga el conjunto de datos procesado del proyecto anterior (dividido en muestras de entrenamiento y pruebas y analizado con el EDA).

#### Paso 2: Construye un boosting

Una forma de optimizar y mejorar los resultados es generar un boosting de tal forma que haya una variedad necesaria que enriquezca la predicción. Entrénalo y analiza sus resultados. Prueba a modificar los hiperparámetros que definen al modelo con distintos valores y analiza su impacto con la precisión final y grafica las conclusiones.

#### Paso 3: Guarda el modelo

Almacena el modelo en la carpeta correspondiente.

#### Paso 4: Analiza y compara los resultados de los modelos

Haz un estudio ahora de los tres modelos utilizados, analiza sus predicciones, la clase con más precisión de predicción y la que menos. ¿Con cuál de los tres modelos te quedas?

> NOTA: Solución: https://github.com/4GeeksAcademy/boosting-algorithms-project-tutorial/blob/main/solution.ipynb