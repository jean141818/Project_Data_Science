# 📊 Predicción de Riesgo Crediticio con Machine Learning: Un Caso Práctico  

Este proyecto explora cómo aplicar modelos de **Machine Learning** para predecir el riesgo crediticio en solicitudes de préstamos hipotecarios. El objetivo es identificar patrones que permitan **predecir la aprobación o rechazo de un préstamo** en función de características socioeconómicas, financieras y demográficas de los solicitantes.  

---

## 📌 Introducción  
El crédito hipotecario es una de las operaciones más importantes dentro del sistema financiero. Predecir si un solicitante tiene altas probabilidades de incumplimiento o aprobación es fundamental para **reducir riesgos** y **mejorar la inclusión financiera**.  

En este caso práctico, trabajamos con un dataset sintético de **5,000 solicitudes de préstamos para vivienda (HomeCreditIQ 5000)** y entrenamos diferentes modelos para evaluar su desempeño en la predicción del estado del préstamo (`Loan_Status`).  

---

## 📂 Dataset  
El conjunto de datos incluye información demográfica, socioeconómica y financiera:  

- `ApplicantIncome`: ingreso principal.  
- `CoapplicantIncome`: ingreso del co-solicitante.  
- `LoanAmount`: monto solicitado.  
- `Loan_Amount_Term`: plazo del préstamo en meses.  
- `Credit_History`: historial crediticio (1 = bueno, 0 = malo).  
- `Dependents, Gender, Education, Property_Area`: variables categóricas.  
- `Loan_Status`: variable objetivo (Y=aprobado, N=rechazado).  

**Objetivo:** predecir `Loan_Status` a partir de las demás variables.  

---

## ⚙️ ETL y Preprocesamiento  
1. Conversión de `Loan_Status` a binario (`Y=1, N=0`).  
2. Transformación de `Dependents` → valores numéricos (ejemplo: `3+ → 3`).  
3. Creación de nueva variable:  
   - `TotalIncome = ApplicantIncome + CoapplicantIncome`.  
4. Limpieza de valores faltantes:  
   - Mediana para numéricos.  
   - Moda para categóricos.  
5. Codificación de variables categóricas (Label Encoding).  
6. Escalado de datos numéricos con `StandardScaler`.  

---

## 🔎 Análisis Exploratorio (EDA)  
- **Distribución de Ingresos:** mayoría concentrada entre **15,000 y 25,000 USD**, simulando población de clase media.  
- **Historial crediticio:** fuerte relación con la aprobación del préstamo.  
- **Zona de propiedad:** solicitudes urbanas presentan mayor tasa de aprobación.  
- **Matriz de correlación:** `ApplicantIncome` y `TotalIncome` están altamente correlacionados (0.84).  

---

## 🤖 Modelado  
Se entrenaron tres modelos principales:  
- **Logistic Regression** (baseline).  
- **Random Forest**.  
- **XGBoost**.  

📏 Métricas evaluadas: **Accuracy, Precision, Recall, F1-score, ROC-AUC**.  

| Modelo              | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 0.705    | 0.705     | 1.000  | 0.827 | 0.500   |
| Random Forest       | 0.693    | 0.707     | 0.965  | 0.816 | 0.504   |
| XGBoost             | 0.662    | 0.708     | 0.885  | 0.786 | -       |

---

## 📈 Interpretación  
- **Logistic Regression** obtuvo el mejor **F1-score (0.827)** → buen balance entre precisión y recall.  
- Logistic y Random Forest lograron **recall > 0.95**, lo que significa que casi todos los préstamos aprobados se identificaron correctamente.  
- El **ROC-AUC cercano a 0.5** muestra que los modelos aún no separan bien las clases (problema de desbalance en los datos).  

---

## ⚠️ Desafío: Desbalance de Clases  
- El dataset contiene **más préstamos aprobados que**
