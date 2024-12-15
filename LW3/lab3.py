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

# Класифікаційна модель
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_class = knn_classifier.predict(X_test)

# Оцінка якості класифікаційної моделі
classification_accuracy = accuracy_score(y_test, y_pred_class)
print("Точність класифікаційної моделі:", classification_accuracy)

# 2. Налаштування оптимальної кількості найближчих сусідів у методі kNN

# 2.1 Знайдіть показник якості моделі kNN на крос-валідації. Подумайте, чи прийнятне
# використання вашої міри (метрики) якості у цій задачі? При необхідності пере-
# рахуйте якість за допомогою іншої метрики з списку.

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

# Створюємо генератор розбиття KFold з 5 блоками, з перемішуванням і фіксованим random_state
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Оцінка якості моделі kNN за допомогою крос-валідації з метрикою точності (accuracy)
accuracy_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='accuracy')
print("Середня точність (accuracy) на крос-валідації:", np.mean(accuracy_scores))

# Якщо необхідно використовувати іншу метрику, наприклад F1-score для незбалансованих класів
f1_scorer = make_scorer(f1_score)
f1_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring=f1_scorer)
print("Середній F1-score на крос-валідації:", np.mean(f1_scores))

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Передбачення ймовірностей для позитивного класу
y_pred_proba = knn_classifier.predict_proba(X_test)[:, 1]

# Обчислення ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC:", roc_auc)

# Обчислення Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)


# 2.2 Здійсніть крос-валідацію моделі при числі сусідів k ∈ [1;50]. Використовуйте
# GridSearchCV . При якому k якість вийшла найкращою? Чому дорівнює ця оцін-
# ка якості? Побудуйте графік значень метрики залежно від k
# ( matplotlib.pyplot.plot() ).

from sklearn.model_selection import GridSearchCV

# Визначимо діапазон значень k
param_grid = {'n_neighbors': np.arange(1, 51)}

# Використовуємо GridSearchCV для пошуку найкращого значення k
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X, y)

# Знаходимо найкраще значення k та відповідну якість
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print("Найкраще значення k:", best_k)
print("Оцінка якості при найкращому k:", best_score)

# Побудуємо графік залежності точності від кількості сусідів k
k_values = np.arange(1, 51)
scores = grid_search.cv_results_['mean_test_score']

plt.plot(k_values, scores, marker='o')
plt.xlabel("Число сусідів (k)")
plt.ylabel("Середня точність (accuracy) на крос-валідації")
plt.title("Залежність точності від числа сусідів k")
plt.show()


# 3 Вибір метрики у методі kNN

from sklearn.model_selection import cross_val_score
import numpy as np

# Створюємо набір значень p від 1 до 10 з 20 рівномірними інтервалами
p_values = np.linspace(1, 10, 10)# зменьшимо до 10 замість 20

# Збережемо результати для кожного p
accuracy_scores = [] # зменьшимо до 3 замість 5

# Перебираємо значення p
for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=p, weights='distance')
    scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy', n_jobs=-1) # n_jobs=-1 - використовує всі ядра процесора для прискорення.
    accuracy_scores.append(scores.mean())

# Знаходимо p з найкращою точністю
best_p = p_values[np.argmax(accuracy_scores)]
best_accuracy = max(accuracy_scores)

print("Найкраще значення p:", best_p)
print("Найвища точність (accuracy):", best_accuracy)

# Побудуємо графік залежності точності від p
plt.plot(p_values, accuracy_scores, marker='o')
plt.xlabel("Значення параметра p")
plt.ylabel("Середня точність (accuracy) на крос-валідації")
plt.title("Залежність точності від параметра p в метриці Мінковського")
plt.show()

# 4 Інші метричні методи

from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score

# Задаємо радіус для моделей RadiusNeighborsClassifier і RadiusNeighborsRegressor
radius = 1.0  

# 1. RadiusNeighborsClassifier
radius_clf = RadiusNeighborsClassifier(radius=radius, weights='distance')
radius_clf_scores = cross_val_score(radius_clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Середня точність RadiusNeighborsClassifier:", radius_clf_scores.mean())

# 2. RadiusNeighborsRegressor (оцінка на основі середньої квадратичної помилки)
radius_reg = RadiusNeighborsRegressor(radius=radius, weights='distance')
radius_reg_scores = cross_val_score(radius_reg, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
print("Середня середньоквадратична помилка RadiusNeighborsRegressor:", -radius_reg_scores.mean())

# 3. NearestCentroid
nearest_centroid_clf = NearestCentroid()
nearest_centroid_scores = cross_val_score(nearest_centroid_clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Середня точність NearestCentroid:", nearest_centroid_scores.mean())