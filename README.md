# Proyecto final — Introducción a Deep Learning

**Estudiante:** Daniel Martinez

Implementación y evaluación de tres arquitecturas entrenadas desde cero: un MLP para datos tabulares, una CNN para imágenes y una LSTM para series temporales. Cada notebook estudia un problema independiente e incluye exploración, preprocesamiento, entrenamiento, evaluación y conclusiones.

## Notebooks

| Modelo | Problema | Notebook | Ejecutar en Colab |
|---|---|---|---|
| MLP | Predecir incumplimiento del siguiente pago de tarjeta | [01_MLP_credito.ipynb](notebooks/01_MLP_credito.ipynb) | [Abrir MLP](https://colab.research.google.com/drive/1pMUROvsOp3jw4y-69vlR4sydTCTmpCtc) |
| CNN | Clasificar hojas de frijol: mancha angular, roya o sanas | [02_CNN.ipynb](notebooks/02_CNN.ipynb) | [Abrir CNN](https://colab.research.google.com/drive/1r7hgf0_DWL8K7WQmbKVqnRt9Ip4O3tAF) |
| LSTM | Pronosticar alquileres de bicicletas de la siguiente hora | [03_RNN.ipynb](notebooks/03_RNN.ipynb) | [Abrir LSTM](https://colab.research.google.com/drive/1sBF8s-xxwizflTEc440L1QjHqlkadFBe) |

Los archivos del repositorio conservan las salidas de las ejecuciones documentadas. Los enlaces de Drive pueden requerir autorización de acceso; también se puede descargar el notebook de GitHub y subirlo a Colab.

## Datasets

### 1. Default of Credit Card Clients — MLP

- [Fuente: UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- 30,000 clientes de Taiwán, 23 variables predictoras, un ID y un objetivo binario.
- El 22.12 % de los clientes incumplió. Se estudian escalas, códigos categóricos y desbalance de clases.
- Particiones: 20,995 clientes de entrenamiento, 3,001 de validación y 6,004 de prueba. Los perfiles con predictores idénticos permanecen en una sola partición.
- Referencia: regresión logística. El preprocesador se ajusta únicamente con entrenamiento.

### 2. Beans — CNN

- [Autores: Makerere AI Lab y NaCRRI](https://github.com/AI-Lab-Makerere/ibean). [Publicación usada en Hugging Face](https://huggingface.co/datasets/AI-Lab-Makerere/beans).
- Fotografías de campo de hojas de frijol en Uganda, con tres clases.
- Particiones publicadas: 1,034 imágenes de entrenamiento, 133 de validación y 128 de prueba.
- Se utiliza Hugging Face porque los enlaces de descarga de Google Storage empleados por TensorFlow Datasets devolvieron HTTP 403.
- La auditoría de píxeles no encontró duplicados exactos dentro ni entre particiones. Esto no descarta fotografías parecidas o relacionadas con la misma planta.
- Redimensionamiento a 96 × 96 RGB y comparación con/sin aumento de datos. Referencia adicional: clase mayoritaria.

### 3. Bike Sharing — LSTM

- [Fuente: UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), registros horarios de Capital Bikeshare de 2011–2012.
- El CSV descargado contiene 17,379 registros. El calendario completo tiene 17,544 horas, con 165 horas sin registro.
- Ventanas de 24 horas consecutivas, sin imputar demandas ausentes. Se exige además disponibilidad del rezago semanal para comparar todos los métodos sobre los mismos objetivos.
- División cronológica del calendario 70/15/15 %. Tras los controles quedan 10,483 ventanas de entrenamiento, 2,632 de validación y 2,435 de prueba.
- Solo se usa demanda, meteorología y calendario del pasado. Se excluyen `casual` y `registered`, cuya suma revela el objetivo `cnt`.
- Referencias: hora anterior, día anterior y semana anterior. El escenario es pronóstico de una hora con actualización del historial, no pronóstico recursivo de todo el periodo.

## Arquitecturas y entrenamiento

| Modelo | Arquitectura | Configuración |
|---|---|---|
| MLP | 91 entradas preparadas → Dense 64 → Dropout 0.2 → Dense 32 → Dropout 0.2 → sigmoid; 8,001 parámetros | Adam 0.001, binary crossentropy, batch 128, máximo 50 épocas, early stopping con paciencia 7 |
| CNN | Tres bloques Conv2D/ReLU de 16, 32 y 64 filtros + MaxPooling → GlobalAveragePooling → Dense 64 → Dropout 0.3 → softmax; 27,939 parámetros | Adam 0.001, sparse categorical crossentropy, batch 32, máximo 30 épocas, paciencia 7 y checkpoints |
| LSTM | 24 pasos × 9 variables → LSTM de 16 o 32 unidades → Dropout 0.2 → Dense 16/ReLU → salida lineal; seleccionada: 5,921 parámetros | Adam 0.001, MSE sobre objetivo estandarizado, batch 64, máximo 25 épocas, paciencia 5 y checkpoints |

Se utiliza semilla 42. Las configuraciones se seleccionan con validación; prueba no se usa para ajustar modelos ni umbrales. No se emplean pesos preentrenados.

## Resultados de prueba

### Clasificación

| Modelo/configuración | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión logística, umbral 0.22 | 0.7753 | 0.4807 | 0.5811 | 0.5262 | 0.7713 |
| **MLP, umbral 0.26** | **0.7798** | **0.4896** | **0.6005** | **0.5394** | **0.7830** |
| **CNN sin aumento** | **0.7578** | **0.7554** | **0.7588** | **0.7562** | — |

En MLP, precision/recall/F1 corresponden a incumplimiento (clase 1); en CNN son promedios macro de las tres clases. Son problemas distintos y sus métricas no constituyen una clasificación entre arquitecturas.

- **MLP:** 774 incumplimientos detectados, 515 no detectados y 807 falsas alarmas. Los umbrales de ambos modelos se eligieron maximizando F1 en validación. Average precision: 0.5563 para MLP y 0.5393 para regresión logística.
- **CNN:** 97 aciertos de 128 imágenes; loss 0.5162. Las variantes con y sin aumento tuvieron F1 macro de validación 0.7784 y 0.7796, respectivamente. Se seleccionó la variante sin aumento; la diferencia es pequeña y no demuestra que el aumento sea perjudicial en general. La referencia constante obtuvo accuracy 0.3359 en prueba.

### Pronóstico horario

| Método | MAE (alquileres/hora) | RMSE (alquileres/hora) | R² |
|---|---:|---:|---:|
| Hora anterior | 82.1622 | 126.1476 | 0.6583 |
| Día anterior | 79.6041 | 131.9135 | 0.6264 |
| Semana anterior | 64.0263 | 107.7723 | 0.7506 |
| **LSTM de 32 unidades** | **31.6875** | **47.9959** | **0.9505** |

La referencia semanal fue la mejor de las referencias en validación. La LSTM redujo su MAE de prueba aproximadamente 50.51 %. Estos resultados corresponden a los 2,435 objetivos elegibles; no representan horas excluidas por falta de historial.

## Instalación y ejecución

### Google Colab — entorno utilizado

1. Abrir un enlace de la tabla o subir a Colab el `.ipynb` descargado de GitHub.
2. Guardar una copia propia si se desea conservar cambios.
3. Usar un entorno Python con CPU; los tres notebooks se ejecutaron con CPU.
4. Ejecutar todas las celdas en orden, desde el inicio. CNN instala `datasets` y `Pillow`; MLP instala `xlrd`.
5. Esperar el entrenamiento y revisar las tablas, figuras y verificaciones finales. Cada notebook es independiente.
6. Guardar el notebook con sus salidas antes de descargarlo o subirlo al repositorio.

Los datos se descargan automáticamente de sus fuentes públicas. No se requieren claves de API ni credenciales de Hugging Face. Si un proveedor cambia sus permisos o enlaces, la descarga puede fallar y deberá revisarse la fuente.

### Entorno local opcional

Crear un entorno virtual de Python compatible con TensorFlow 2.21 e instalar:

```bash
python -m pip install -r requirements.txt
python -m notebook
```

Abrir un archivo dentro de `notebooks/` y ejecutar en orden. La validación documentada corresponde a Colab, no a una instalación local nueva. Los notebooks escriben en `/content/resultados_mlp`, `/content/resultados_cnn` y `/content/resultados_rnn`; para usarlos fuera de Colab hay que sustituir esas rutas por carpetas locales con permiso de escritura.

`requirements.txt` fija las versiones observadas de las bibliotecas principales y rangos para otras dependencias; no es un archivo de bloqueo completo. CNN y LSTM generan, además, archivos con las versiones exactas de sus dependencias durante la ejecución. Hardware, versiones y aleatoriedad pueden producir variaciones entre ejecuciones.

## Resultados guardados

Los notebooks conservan código, tablas y figuras. Al ejecutarlos también generan modelos `.keras`, preprocesadores/escaladores, métricas CSV y configuración. CNN y LSTM conservan checkpoints del mejor estado por pérdida de validación. Los tres verifican la recarga del modelo; LSTM comprueba también sus dos escaladores.

**Los archivos de `/content/` son temporales de Colab y no quedan incluidos automáticamente en el `.ipynb` ni en GitHub.** Para conservar los modelos entrenados fuera de la sesión deben descargarse o copiarse a un almacenamiento persistente. El repositorio permite volver a generarlos ejecutando los notebooks.

## Conclusiones y aprendizajes

- En clasificación desbalanceada, accuracy puede ocultar baja detección de la clase minoritaria. El umbral implica un compromiso entre recall y falsas alarmas.
- El MLP mejoró modestamente sobre regresión logística; mayor complejidad no garantiza una mejora grande.
- En CNN, el aumento de datos debe evaluarse con validación en lugar de asumir que siempre mejora resultados.
- En series temporales, detectar huecos, respetar la cronología y usar referencias estacionales es tan importante como definir la arquitectura recurrente.
- El escalado ajustado solo con entrenamiento, los checkpoints y las comprobaciones de recarga ayudan a evitar errores de evaluación y a recuperar el trabajo.

## Limitaciones

Se documenta una ejecución por configuración, con una sola semilla y partición. No se demuestra significancia estadística ni generalización a nuevas poblaciones, fincas, ciudades o periodos. La CNN tiene solo 128 imágenes de prueba; la LSTM excluye objetivos con historia incompleta y sus errores pueden estar correlacionados. Los datos crediticios son históricos y no se evaluaron equidad ni costos financieros reales. Los proyectos son académicos y no son herramientas de decisión validadas.

## Fuentes y atribución

- Yeh, I-C. (2009). *Default of Credit Card Clients*. UCI. [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H). Licencia CC BY 4.0.
- Makerere AI Lab / NaCRRI (2020). *Beans*. [Repositorio y licencia MIT](https://github.com/AI-Lab-Makerere/ibean).
- Fanaee-T, H. (2013). *Bike Sharing*. UCI. [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). Licencia CC BY 4.0.

Los datos originales pertenecen a sus respectivos autores. El procesamiento, entrenamiento y evaluación utilizados en este proyecto se describen en los notebooks.
