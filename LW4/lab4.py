import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

sns.set(rc={'figure.figsize':(10, 8)})

data = pd.read_csv('./data/bank-additional-full.csv', delimiter = ';')

# Видалимо колонку 'duration'
data = data.drop(columns=['duration'])

# Закодуємо категоріальні колонки
label_encoder = LabelEncoder()

# Перетворимо бінарні категоріальні ознаки 'default', 'housing', 'loan', 'y' у числові
for col in ['default', 'housing', 'loan', 'y']:
    data[col] = label_encoder.fit_transform(data[col])

# Закодуємо інші категоріальні ознаки за допомогою one-hot кодування
data = pd.get_dummies(data, drop_first=True)

# Відокремлюємо ознаки (X) та цільову змінну (y)
X = data.drop(columns=['y'])
y = data['y']

# Розділимо дані на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Стандартизуємо  ознаки
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Ініціалізація моделі
clf = DecisionTreeClassifier(random_state=42)

# Навчання моделі
clf.fit(X_train, y_train)

# Прогнозування
y_pred = clf.predict(X_test)

# Оцінка точності
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність класифікації (accuracy): {accuracy:.4f}")


# 2 Налаштування гіперпараметрів дерева

# 2.1 2.2

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

# Створення генератора розбиття
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Модель
clf = DecisionTreeClassifier(random_state=42)

# Сітка гіперпараметрів
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# GridSearchCV для підбору параметрів
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring=make_scorer(accuracy_score),
    cv=kf,
    n_jobs=-1
)

# Навчання з пошуком параметрів
grid_search.fit(X_train, y_train)

# Найкращі параметри
print("Найкращі гіперпараметри:", grid_search.best_params_)

# Оцінка на тестових даних
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність на тестових даних: {accuracy:.4f}")

# 2.3 

from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Функція для побудови валідаційної кривої
def plot_validation_curve(param_name, param_range, X, y, clf, cv):
    train_scores = []
    test_scores = []

    # Проходимо по значеннях гіперпараметра
    for param in param_range:
        params = {param_name: param}
        clf.set_params(**params)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        test_scores.append(np.mean(scores))
    
    # Побудова графіка
    plt.plot(param_range, test_scores, marker='o', label='Validation Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Validation Curve for {param_name}')
    plt.grid()
    plt.legend()
    plt.show()

# Гіперпараметри для перевірки
max_depth_range = [1, 2, 3, 5, 10, 20, None]
min_samples_split_range = [2, 5, 10, 20]
min_samples_leaf_range = [1, 2, 4, 10]

# Крос-валідація
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Модель
clf = DecisionTreeClassifier(random_state=42)

# Побудова кривих для max_depth
plot_validation_curve('max_depth', max_depth_range, X_train, y_train, clf, kf)

# Побудова кривих для min_samples_split
plot_validation_curve('min_samples_split', min_samples_split_range, X_train, y_train, clf, kf)

# Побудова кривих для min_samples_leaf
plot_validation_curve('min_samples_leaf', min_samples_leaf_range, X_train, y_train, clf, kf)


# 2.4 2.5

from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# Візуалізація дерева
plt.figure(figsize=(12, 8))
plot_tree(best_model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Графічне зображення дерева рішень")
plt.show()

# Оцінка важливості ознак
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Показати важливість ознак
import seaborn as sns
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Важливість ознак (feature_importances_)")
plt.xlabel("Важливість")
plt.ylabel("Ознаки")
plt.show()

# Показати важливість ознак для аналізу
feature_importances


# 3.1

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Створення та навчання моделі випадкового лісу
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred_rf = rf_model.predict(X_test)

# Оцінка точності моделі
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf, target_names=['No', 'Yes'])

print(f"Точність моделі випадкового лісу: {rf_accuracy:.4f}")
print("\nЗвіт класифікації:")
print(rf_report)


# 3.2

# Оптимізована сітка параметрів для підбору (зменшена кількість комбінацій)
param_grid_optimized = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None]
}

# Ініціалізація GridSearchCV з оптимізованою сіткою
grid_search_optimized = GridSearchCV(estimator=rf, param_grid=param_grid_optimized, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

# Навчання моделі з підбором параметрів
grid_search_optimized.fit(X_train, y_train)

# Найкращі параметри та результати
best_params_optimized = grid_search_optimized.best_params_
best_score_optimized = grid_search_optimized.best_score_

# Прогноз на тестових даних з найкращою моделлю
best_model_optimized = grid_search_optimized.best_estimator_
y_pred_best_rf_optimized = best_model_optimized.predict(X_test)
best_rf_accuracy_optimized = accuracy_score(y_test, y_pred_best_rf_optimized)
best_rf_report_optimized = classification_report(y_test, y_pred_best_rf_optimized, target_names=['No', 'Yes'])

# Виведення результатів
print(f"Найкращі параметри (оптимізована сітка): {best_params_optimized}")
print(f"Точність на валідаційних даних (GridSearchCV): {best_score_optimized:.4f}")
print(f"Точність на тестових даних: {best_rf_accuracy_optimized:.4f}")
print("\nЗвіт класифікації для найкращої моделі (оптимізована сітка):")
print(best_rf_report_optimized)


# 3.3

import matplotlib.pyplot as plt
import numpy as np

# Функція для побудови валідаційних кривих
def plot_validation_curve(param_name, param_range, X_train, y_train, model, cv):
    from sklearn.model_selection import cross_val_score
    train_scores = []
    validation_scores = []

    for param in param_range:
        if param_name == 'max_features' and param == 'None':
            params = {param_name: None}  # Передаємо None у модель, якщо задано 'None'
        else:
            params = {param_name: param}
        model.set_params(**params)
        # params = {param_name: param}
        # model.set_params(**params)
        # Крос-валідація
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        validation_scores.append(np.mean(scores))

    # Побудова графіку
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, validation_scores, marker='o', label='Validation Accuracy')
    plt.title(f'Validation Curve for {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.show()


# Побудова валідаційних кривих для кожного гіперпараметра
plot_validation_curve('n_estimators', [10, 50, 100, 200], X_train, y_train, rf, cv=3)
plot_validation_curve('max_depth', [5, 10, 15, None], X_train, y_train, rf, cv=3)
plot_validation_curve('min_samples_split', [2, 5, 10], X_train, y_train, rf, cv=3)
plot_validation_curve('min_samples_leaf', [1, 2, 4], X_train, y_train, rf, cv=3)
plot_validation_curve('max_features', ['sqrt', 'log2', 'None'], X_train, y_train, rf, cv=3)


# 3.4

# Оцінка важливості ознак моделі Random Forest
feature_importances_rf = pd.Series(rf_model.feature_importances_, index=X.columns)

# Сортування за важливістю та вибір топ-10 ознак
top_10_features_rf = feature_importances_rf.sort_values(ascending=False).head(10)

# Побудова стовпчастої діаграми
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_features_rf, y=top_10_features_rf.index, hue=top_10_features_rf.index, dodge=False, legend=False, palette="viridis")
plt.title("Топ-10 найважливіших ознак моделі Random Forest")
plt.xlabel("Важливість ознаки")
plt.ylabel("Ознаки")
plt.grid(axis='x')
plt.show()