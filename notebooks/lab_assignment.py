# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
#     jupytext_version: 1.14.5
# ---

# %% [markdown]
# # Практическое задание: Pipeline и GridSearchCV
# 
# В этом задании мы:
# 1. Загрузим датасет (например, Titanic).
# 2. Разделим данные на train/test.
# 3. Реализуем кастомный трансформер для подсчета пропусков.
# 4. Создадим Pipeline для обработки числовых и категориальных признаков, а также обучения модели LogisticRegression.
# 5. Проведем GridSearchCV для подбора гиперпараметров.
# 6. Оценим качество на тестовой выборке и выведем лучшие параметры.

# %%
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report

# %% [markdown]
# ## 1. Загрузка данных (Titanic)

# %%
# Загрузим датасет Titanic с помощью seaborn
df = sns.load_dataset('titanic')

# Выберем подмножество признаков для демонстрации (числовые и категориальные)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'alone']
target = 'survived'

X = df[features]
y = df[target]

print("Размер датасет:", X.shape)
X.head()

# %% [markdown]
# ## 2. Разделение данных на train/test

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# %% [markdown]
# ## 3. Кастомный трансформер для подсчета пропусков в строке

# %%
class MissingValuesCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Создаем копию, чтобы не модифицировать исходный датафрейм
        X_out = X.copy()
        # Считаем количество пропусков по строкам
        if isinstance(X_out, pd.DataFrame):
            missing_counts = X_out.isnull().sum(axis=1).values.reshape(-1, 1)
        else:
            missing_counts = np.isnan(X_out).sum(axis=1).reshape(-1, 1)
        
        # Добавляем новый признак в виде DataFrame или numpy массива
        # В рамках Pipeline удобнее возвращать DataFrame или numpy array. 
        # Здесь мы возвращаем исходные данные с добавленной колонкой или просто возвращаем сам новый признак,
        # но по условию нам нужно добавить его в пайплайн. 
        # Проще всего реализовать это через FeatureUnion или ColumnTransformer, либо возвращать pandas DataFrame.
        # Для простоты вернем исходные данные, превратив в DataFrame, и добавим колонку.
        pass

# Более элегантный способ через ColumnTransformer — передавать весь DataFrame, 
# но проще сделать кастомный трансформер, который добавляет колонку к датафрейму, 
# либо использовать FeatureUnion. 
# Давайте напишем трансформер, который добавляет новый столбец к DataFrame:

class AddMissingCountFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            X_copy['n_missing'] = X_copy.isnull().sum(axis=1)
            return X_copy
        else:
            X_df = pd.DataFrame(X)
            X_df['n_missing'] = X_df.isnull().sum(axis=1)
            return X_df

# %% [markdown]
# ## 4. Создание Pipeline

# %%
# Определим числовые и категориальные колонки с учетом нового признака
# Сначала применим наш кастомный трансформер ко всему датасету, 
# а затем разведем числовые и категориальные колонки (включая 'n_missing').

from sklearn.pipeline import FeatureUnion

numeric_features = ['age', 'sibsp', 'parch', 'fare', 'n_missing']
categorical_features = ['pclass', 'sex', 'embarked', 'class', 'alone']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Полный pipeline
pipeline = Pipeline(steps=[
    ('add_missing_flag', AddMissingCountFeature()),
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# %% [markdown]
# ## 5. GridSearchCV для подбора гиперпараметров

# %%
param_grid = {
    'preprocessor__num__imputer__strategy': ['median', 'mean'],
    'classifier__C': [0.01, 0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# %% [markdown]
# ## 6. Оценка качества и результаты

# %%
print("Best parameters found:")
print(grid_search.best_params_)

print(f"\nBest CV Accuracy: {grid_search.best_score_:.4f}")

# Оценка на тестовой выборке
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
