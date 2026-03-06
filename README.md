### ML_NASA-Exoplanets_Detection_Method
Proyecto : Machine Learning:
Clasificación del método de detección (detection_method)

Descripción breve, Objetivo y planteamiento:
Predecir el detection_method (método de detección) a partir de características del exoplaneta y su sistema utilizando dataset dirigido de Kaggle NASA Exoplanets (CSV en src/data_sample/). El utilizado para ayudar a entender qué variables están asociadas a cada método y qué sesgos hay en el catálogo (por ejemplo, algunos métodos detectan mejor planetas con periodos cortos). Resolveremos la problema de clasificación supervisada multiclase.

Métrica principal: usaremos F1-macro porque hay desbalanceo entre clases (no queremos que el modelo solo acierte la clase mayoritaria).
Elegimos detection_method porque representa la técnica con la que se detectó el planeta (tránsito, velocidad radial, etc.).
Esto nos permite ver si las variables del sistema contienen “señal” asociada a cada método y entender sesgos del catálogo.

También filtramos clases con muy pocos ejemplos para no entrenar con clases casi vacías. target_col = "detection_method" X, y = make_xy(df_clean, target_col=target_col, min_class=25)

División train/test
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)

EDA dirigido solo al TRAIN
Distribución del target
Missing values
Histogramas numéricas / conteos categóricas
Correlaciones numéricas
Relación de algunas variables con el target eda_basic(X_train, y_train, target_name=target_col, img_dir="src/img")
Conclusiones del EDA
El dataset tiene un tamaño suficiente para entrenar un modelo (miles de filas) y combina variables numéricas (masa, radio, periodo orbital, etc.) y categóricas (por ejemplo unidades o categorías).
Los valores missing eran bajos en general (la columna con más missing rondaba ~5%), por lo que no era necesario eliminar columnas por falta de datos. Aun así, se revisó el porcentaje de missing y se decidió imputar en el pipeline para no perder información.
La variable objetivo detection_method está muy desbalanceada: hay un método claramente mayoritario (Transit) y varios minoritarios (p. ej. Direct Imaging). Esto explica por qué la accuracy puede ser engañosa y por qué usamos F1-macro como métrica principal.
La columna name se comporta como un identificador (muchísimos valores únicos), así que se eliminó en la reducción preliminar porque no ayuda a generalizar y podría hacer que el modelo memorice.
En la matriz de correlación se observan relaciones entre algunas variables numéricas, lo que justifica revisar redundancia y mantener un preprocesado robusto (imputación + escalado para numéricas y one-hot para categóricas).
En los boxplots/histogramas aparecen valores extremos en algunas variables numéricas, por lo que se revisaron outliers y se aplicó un tratamiento sencillo (clipping) para evitar que dominen el entrenamiento.
Reducción preliminar de variables
Quitamos cosas claramente poco útiles:

constantes
columnas tipo ID (casi únicas)
demasiados missings
categóricas con demasiados niveles X_train_red, X_test_red, dropped_cols = preliminary_feature_reduction(X_train, X_test) print("Columnas eliminadas:", dropped_cols)
Outliers y relevancia de variables del TRAIN
Aquí detectamos valores extremos en variables numéricas y aplicamos un tratamiento sencillo (clipping por IQR). También sacamos un ranking simple de features más relevantes para el target usando Mutual Information. outlier_report(X_train_red) X_train_red, X_test_red = clip_outliers_iqr(X_train_red, X_test_red) mi_rank = feature_relevance_mi(X_train_red, y_train, top=10)

Modelos baseline + validación cruzada (CV)
Probamos varios modelos con CV y escogemos el mejor por F1-macro:

Dummy (baseline)
Logistic Regression
Random Forest
Gradient Boosting pre = build_preprocessor(X_train_red) pipes, leaderboard, best_name = compare_models_cv(X_train_red, y_train, pre, seed=42) leaderboard
Búsqueda de hiperparámetros (RandomizedSearchCV):
Ajustamos hiperparámetros del mejor modelo usando CV y optimizando F1-macro if best_name == "Dummy": best_name = "LogReg" best_model, best_params, best_cv = tune_model(best_name, pipes[best_name], X_train_red, y_train, seed=42) print(best_params, best_cv)

Evaluación final en TEST
Evaluamos el modelo final en el conjunto de test (classification report + matriz de confusión). preds = evaluate_model(best_model, X_test_red, y_test, img_dir="src/img/conf_matrix_test.png")

Conclusiones de la evaluación del modelo
La matriz de confusión confirma que el modelo acierta muy bien la clase mayoritaria (Transit) y también funciona bien con Radial Velocity.

Los errores más comunes se concentran en las clases con menos ejemplos (por ejemplo Direct Imaging), donde es más fácil confundirlas con otras porque hay pocos casos para aprender patrones estables.

Esto es coherente con el desbalanceo: aunque el rendimiento global sea muy alto, las clases minoritarias tienen más riesgo de bajar en recall (se “escapan” ejemplos que deberían ser de esa clase).

Para mejorar específicamente esas clases, se podría:

usar class_weight="balanced" en algunos modelos,
hacer re-muestreo (oversampling),
o agrupar clases raras si el objetivo del proyecto lo permite.
El baseline (Dummy) consigue una accuracy relativamente alta porque predice casi siempre la clase mayoritaria, pero su F1-macro es bajo, así que no es un modelo útil para todas las clases.

Al comparar modelos con validación cruzada, los modelos de árboles (Random Forest / Gradient Boosting) funcionan mucho mejor que una regresión logística, lo que sugiere que hay relaciones no lineales importantes entre variables.

Con tuning de hiperparámetros (RandomizedSearchCV) se consigue un modelo final con resultados muy altos en test (accuracy cercana a 0.98 y F1-macro alta), lo que indica que el método de detección es muy predecible a partir de las variables disponibles.

La métrica principal usada es F1-macro porque hay desbalanceo: así evitamos “premiar” a un modelo que solo acierta Transit y falla el resto.

Aun con buen rendimiento global, el modelo puede estar aprovechando sesgos del catálogo (qué planetas suelen encontrarse con cada técnica). Esto no es malo para el objetivo del proyecto, pero conviene mencionarlo como limitación.

Guardado del modelo y resultados
info = { "target": target_col, "best_model": best_name, "best_params": best_params, "best_cv_f1_macro": best_cv, "train_shape": [int(X_train_red.shape[0]), int(X_train_red.shape[1])], "test_shape": [int(X_test_red.shape[0]), int(X_test_red.shape[1])], }

save_model_and_card( best_model, model_path="src/models/best_model_detection_method.joblib", card_path="src/models/model_card_detection_method.json", info=info )

INSTRUCCIONES (CREAR EL ENVIRONMENT DESDE CERO Y EJECUTAR EN VS CODE)
Python recomendado: 3.11

Requisitos
Tener instalado Python 3.11
Tener VS Code
Instalar extensiones en VS Code: "Python" y "Jupyter"
Abrir el proyecto
Abre VS Code
File > Open Folder...
Selecciona la carpeta del proyecto
Crear el entorno virtual (.venv) Abre la terminal en VS Code (Terminal > New Terminal) y ejecuta:
Windows: py -3.11 -m venv .venv

macOS / Linux: python3.11 -m venv .venv

Activar el entorno Windows (PowerShell): ..venv\Scripts\Activate.ps1
Windows (cmd): ..venv\Scripts\activate.bat

macOS / Linux: source .venv/bin/activate

(IMPORTANTE: si en la terminal no aparece "(.venv)" al inicio, NO está activado)

Instalar dependencias Con el entorno activado:
python -m pip install --upgrade pip pip install -r requirements.txt

Registrar el kernel para Jupyter (una vez) Con el entorno activado:
python -m ipykernel install --user --name exoplanets-311 --display-name "Python 3.11 (exoplanets)"

Seleccionar kernel en VS Code
Abre el notebook (.ipynb)
Arriba a la derecha: "Select Kernel"
Elige: "Python 3.11 (exoplanets)"
Ejecutar
En el notebook pulsa "Run All" o ejecuta celdas una a una
Conclusiones finales del proyecto
Se ha planteado un problema claro: predecir el método de detección (detection_method) a partir de características del exoplaneta y su sistema.

Tiene sentido usar Machine Learning porque la relación entre variables físicas/observacionales y el método de detección no es trivial, y además es un problema de clasificación multiclase con desbalanceo.

Se siguió un flujo completo: carga y limpieza, split train/test, EDA dirigido, reducción preliminar de features, pipeline de preprocesado, comparación de modelos con validación cruzada, búsqueda de hiperparámetros y evaluación final en test.

El modelo final obtiene muy buen rendimiento y muestra que el dataset contiene suficiente información para diferenciar métodos de detección.

Como trabajo futuro, se podría:

repetir el mismo pipeline para predecir otro target (por ejemplo planet_type o estimar masa/radio),
mejorar el tratamiento de clases minoritarias,
y analizar interpretabilidad (importancias / SHAP) para entender qué variables influyen más en cada método.
Autores: Carmen Gomez Garcia Marta Estevez Neha Malhotra
