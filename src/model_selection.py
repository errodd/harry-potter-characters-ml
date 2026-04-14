import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -------------------------------
# 1. CONFIGURACIÓN INICIAL
# -------------------------------

# Crear carpeta para guardar modelos y resultados
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# -------------------------------
# 2. CARGAR DATOS PROCESADOS
# -------------------------------

print("Cargando datos procesados...")
try:
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv')
    print("Datos cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Asegúrate de haber ejecutado el script de procesamiento de datos primero.")
    exit()

# Aplanar y_train/y_test si tienen una sola columna (convertir a Series)
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values.ravel()
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.values.ravel()

print(f"Dimensiones de entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Dimensiones de prueba: {X_test.shape}, {y_test.shape}")

# -------------------------------
# 3. DEFINIR MODELOS A EVALUAR
# -------------------------------

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# -------------------------------
# 4. EVALUACIÓN CON VALIDACIÓN CRUZADA (CV = 5)
# -------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\nEvaluando modelos con validación cruzada (5-fold)...")
for name, model in models.items():
    # Calcular accuracy promedio con CV
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    
    # Entrenar en todo el conjunto de entrenamiento para evaluación en test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas en test
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_test = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_test = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'Modelo': name,
        'CV Mean Accuracy': mean_cv,
        'CV Std': std_cv,
        'Test Accuracy': acc_test,
        'Test Precision': prec_test,
        'Test Recall': rec_test,
        'Test F1': f1_test
    })
    
    print(f"{name:20} -> CV Accuracy: {mean_cv:.4f} (+/- {std_cv:.4f}) | Test Accuracy: {acc_test:.4f}")

# -------------------------------
# 5. COMPARACIÓN DE RESULTADOS
# -------------------------------

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Test Accuracy', ascending=False)
print("\nTabla comparativa de modelos:")
print(df_results.to_string(index=False))

# Guardar resultados en CSV
df_results.to_csv('reports/model_comparison.csv', index=False)

# -------------------------------
# 6. SELECCIONAR EL MEJOR MODELO (basado en Test Accuracy)
# -------------------------------

best_model_name = df_results.iloc[0]['Modelo']
best_model = models[best_model_name]
print(f"\nMejor modelo: {best_model_name}")

# Guardar el modelo entrenado
joblib.dump(best_model, f'models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
print(f"Modelo guardado en 'models/best_model_{best_model_name.replace(' ', '_').lower()}.pkl'")

# -------------------------------
# 7. MATRIZ DE CONFUSIÓN DEL MEJOR MODELO
# -------------------------------

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusión - {best_model_name}')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('reports/confusion_matrix_best_model.png', dpi=150)
plt.show()

# -------------------------------
# 8. REPORTE DE CLASIFICACIÓN DETALLADO
# -------------------------------

print("\n Reporte de clasificación del mejor modelo:")
print(classification_report(y_test, y_pred_best, zero_division=0))

# Guardar reporte en archivo de texto
with open('reports/classification_report_best_model.txt', 'w') as f:
    f.write(f"Modelo: {best_model_name}\n\n")
    f.write(classification_report(y_test, y_pred_best, zero_division=0))

print("\nProceso de selección de modelos completado.")