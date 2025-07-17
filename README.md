# 📘 Trabajo de Fin de Máster (TFM) – Aplicación de Técnicas de Machine Learning para la Predicción de Diferencias de Color Percibidas

Este repositorio contiene el código y documentación asociados al Trabajo de Fin de Máster titulado **"<Título del Proyecto>"**, realizado por **Pablo Gallardo** para la obtención del título de **Máster en en Inteligencia de Negocio y Big Data en Entornos Seguros** por la **Universidad de Valladolid**.

## 📄 Resumen

> Este trabajo explora el uso de técnicas de *machine learning* para la predicción de las diferencias de color. Se desarrolla un sistema basado en Python que recopila datos, entrena modelos supervisados y visualiza los resultados mediante distintas gráficas.

---

## 🧠 Objetivos

- Evaluar el rendimiento de distintos modelos de machine learning y redes neuronales a la hora de predecir diferencias de color
- Evaluar si algun modelo mejora el valor de STRESS de CIEDE2000.
- <Validación experimental o de resultados>
- <Aplicación práctica o despliegue, si aplica>

---

## 🛠️ Tecnologías y Herramientas

- Lenguaje principal: `Python`
- Librerías: `Scikit-learn`, `Pandas`, `NumPy`, `Matplotlib`, `PyTorch`/`TensorFlow`, `XGBoost`.
- Base de datos: `Excel`

---

## 📁 Estructura del Proyecto

```bash
📦 tfm-proyecto
├── bbdd/                # Datos brutos
├── notebooks/           # Notebooks de análisis y prototipado
├── requirements.txt     # Dependencias
├── README.md            # Este archivo
```

## 🚀 Ejecución del proyecto

1. Clonar el repositorio

2. Instalar las dependencias
```bash
python -m venv venv
venv\Scripts\activate  # En Mac: source venv/bin/activate 
pip install -r requirements.txt
```
3. Ejecutar el entrenamiento concreto y ver sus resultados
```bash
Los modelos finales del TFM son los de la carpeta train_validate_test.
python .\train_validate_test\entrenamiento_train_validate_test_pytorch.py
```
