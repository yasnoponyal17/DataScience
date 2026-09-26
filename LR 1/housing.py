# Основные библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Библиотеки для машинного обучения
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Дополнительные библиотеки
from scipy import stats
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
# Настройки для графиков
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
print("Все библиотеки успешно импортированы!")

df = pd.read_csv('train.csv')
print("Данные успешно загружены!")
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

# Общая информация о датасете
print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ:")
print("="*50)
print(f"Размер: {df.shape}")
print(f"Количество числовых признаков: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"Количество категориальных признаков: {df.select_dtypes(include=['object']).shape[1]}")

# Информация о типах данных
print(f"\nТипы данных:")
print(df.dtypes.value_counts())

# Базовая статистика по целевой переменной
print(f"\nСтатистика по целевой переменной (SalePrice):")
print(f"Среднее: ${df['SalePrice'].mean():,.2f}")
print(f"Медиана: ${df['SalePrice'].median():,.2f}")
print(f"Стандартное отклонение: ${df['SalePrice'].std():,.2f}")
print(f"Минимум: ${df['SalePrice'].min():,.2f}")
print(f"Максимум: ${df['SalePrice'].max():,.2f}")

# 2.2. Анализ пропущенных значений

# Подробный анализ пропущенных значений
def analyze_missing_values(dataframe):
    """Анализирует пропущенные значения в датасете"""
    missing_data = dataframe.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    missing_percent = (missing_data / len(dataframe)) * 100
    
    missing_df = pd.DataFrame({
        'Количество пропусков': missing_data,
        'Процент пропусков': missing_percent
    })
    
    return missing_df

missing_analysis = analyze_missing_values(df)
print("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
print("="*50)

if len(missing_analysis) > 0:
    print(missing_analysis)
    
    # Визуализация пропусков
    plt.figure(figsize=(12, 8))
    
    # График 1: Количество пропусков
    plt.subplot(2, 1, 1)
    missing_analysis['Количество пропусков'].plot(kind='bar', color='coral')
    plt.title('Количество пропущенных значений по признакам')
    plt.ylabel('Количество пропусков')
    plt.xticks(rotation=45)
    
    # График 2: Процент пропусков
    plt.subplot(2, 1, 2)
    missing_analysis['Процент пропусков'].plot(kind='bar', color='lightblue')
    plt.title('Процент пропущенных значений по признакам')
    plt.ylabel('Процент пропусков (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nОбщее количество признаков с пропусками: {len(missing_analysis)}")
    print(f"Общий процент пропусков в датасете: {df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.2f}%")
else:
    print("Пропущенных значений не найдено!")

# 2.3. Категоризация признаков

# Разделяем признаки на категории для лучшего понимания
def categorize_features(dataframe):
    """Категоризирует признаки по типам"""
    # Числовые признаки
    numeric_features = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numeric_features:
        numeric_features.remove('Id')
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')
    
    # Категориальные признаки
    categorical_features = dataframe.select_dtypes(include=['object']).columns.tolist()
    
    # Разделяем числовые на дискретные и непрерывные
    discrete_features = []
    continuous_features = []
    
    for feature in numeric_features:
        unique_values = dataframe[feature].nunique()
        if unique_values < 20:  # Эвристика для определения дискретных признаков
            discrete_features.append(feature)
        else:
            continuous_features.append(feature)
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'discrete': discrete_features,
        'continuous': continuous_features
    }

feature_categories = categorize_features(df)

print("КАТЕГОРИЗАЦИЯ ПРИЗНАКОВ:")
print("="*50)
print(f"Всего признаков (без Id и SalePrice): {len(feature_categories['numeric']) + len(feature_categories['categorical'])}")
print(f"\nЧисловые признаки ({len(feature_categories['numeric'])}):")
print(f"  - Дискретные ({len(feature_categories['discrete'])}): {feature_categories['discrete']}")
print(f"  - Непрерывные ({len(feature_categories['continuous'])}): {feature_categories['continuous']}")
print(f"\nКатегориальные признаки ({len(feature_categories['categorical'])}):")
print(feature_categories['categorical'])

# Шаг 3: Exploratory Data Analysis (EDA)
# 3.1. Анализ целевой переменной

# Подробный анализ целевой переменной
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Гистограмма
axes[0, 0].hist(df['SalePrice'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Распределение цен на дома')
axes[0, 0].set_xlabel('Цена ($)')
axes[0, 0].set_ylabel('Частота')

# Ящик с усами
axes[0, 1].boxplot(df['SalePrice'])
axes[0, 1].set_title('Ящик с усами для цен')
axes[0, 1].set_ylabel('Цена ($)')

# Логарифмическое распределение
log_prices = np.log(df['SalePrice'])
axes[1, 0].hist(log_prices, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Логарифмическое распределение цен')
axes[1, 0].set_xlabel('log(Цена)')
axes[1, 0].set_ylabel('Частота')

# Q-Q plot для проверки нормальности
stats.probplot(df['SalePrice'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q график для цен')

plt.tight_layout()
plt.show()

# Статистический анализ распределения
skewness = skew(df['SalePrice'])
print(f"Коэффициент асимметрии (skewness): {skewness:.3f}")

if skewness > 1:
    print("Распределение сильно смещено вправо")
elif skewness > 0.5:
    print("Распределение умеренно смещено вправо")
elif skewness < -1:
    print("Распределение сильно смещено влево")
elif skewness < -0.5:
    print("Распределение умеренно смещено влево")
else:
    print("Распределение близко к нормальному")

# 3.2. Анализ числовых признаков

# Корреляционный анализ числовых признаков
numeric_df = df[feature_categories['numeric'] + ['SalePrice']]
correlation_matrix = numeric_df.corr()

# Находим признаки с высокой корреляцией с целевой переменной
target_correlations = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)
high_corr_features = target_correlations[target_correlations > 0.3].index.tolist()
high_corr_features.remove('SalePrice')

print("ПРИЗНАКИ С ВЫСОКОЙ КОРРЕЛЯЦИЕЙ С ЦЕНОЙ (>0.3):")
print("="*50)
for feature in high_corr_features:
    corr_value = correlation_matrix.loc[feature, 'SalePrice']
    print(f"{feature}: {corr_value:.3f}")

# Визуализация корреляций
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.show()

# Диаграммы рассеяния для самых важных признаков
if len(high_corr_features) >= 4:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(high_corr_features[:4]):
        axes[i].scatter(df[feature], df['SalePrice'], alpha=0.6)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('SalePrice')
        axes[i].set_title(f'Зависимость цены от {feature}')
        
        # Добавляем линию тренда
        z = np.polyfit(df[feature].dropna(), df['SalePrice'][df[feature].notna()], 1)
        p = np.poly1d(z)
        axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()

# 3.3. Анализ категориальных признаков

# Анализ категориальных признаков
def analyze_categorical_feature(dataframe, feature, target='SalePrice', top_n=10):
    """Анализирует категориальный признак"""
    # Подсчет количества категорий
    value_counts = dataframe[feature].value_counts()
    
    # Средняя цена по категориям
    mean_prices = dataframe.groupby(feature)[target].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
    
    print(f"\nАНАЛИЗ ПРИЗНАКА '{feature}':")
    print("-" * 40)
    print(f"Количество уникальных значений: {dataframe[feature].nunique()}")
    print(f"Топ-{min(top_n, len(mean_prices))} категорий по средней цене:")
    print(mean_prices.head(top_n))
    
    return mean_prices

# Анализируем несколько ключевых категориальных признаков
key_categorical = ['Neighborhood', 'OverallQual', 'KitchenQual', 'GarageType']

for feature in key_categorical:
    if feature in df.columns:
        analyze_categorical_feature(df, feature)

# Визуализация для самых важных категориальных признаков
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, feature in enumerate(key_categorical[:4]):
    if feature in df.columns:
        # Ящики с усами
        df.boxplot(column='SalePrice', by=feature, ax=axes[i])
        axes[i].set_title(f'Распределение цен по {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('SalePrice')
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# Шаг 4: Обработка пропущенных значений
# 4.1. Стратегии обработки пропусков

# Создаем копию данных для обработки
df_processed = df.copy()

def handle_missing_values(dataframe):
    """Обрабатывает пропущенные значения с учетом специфики признаков"""
    
    print("ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
    print("="*50)
    
    # Для некоторых признаков пропуски означают отсутствие объекта
    # Например, отсутствие гаража, подвала, камина и т.д.
    
    categorical_na_as_none = ['Alley', 'MasVnrType', 'BsmtQual', 'GarageType']
    
    for feature in categorical_na_as_none:
        if feature in dataframe.columns:
            before_count = dataframe[feature].isnull().sum()
            dataframe[feature].fillna('None', inplace=True)
            print(f"{feature}: заполнено {before_count} пропусков значением 'None'")
    
    # Для числовых признаков, связанных с отсутствующими объектами
    numeric_na_as_zero = ['MasVnrArea']
    
    for feature in numeric_na_as_zero:
        if feature in dataframe.columns:
            before_count = dataframe[feature].isnull().sum()
            dataframe[feature].fillna(0, inplace=True)
            print(f"{feature}: заполнено {before_count} пропусков нулями")
    
    # Для остальных числовых признаков используем медиану
    numeric_features = dataframe.select_dtypes(include=[np.number]).columns
    for feature in numeric_features:
        if dataframe[feature].isnull().sum() > 0:
            median_value = dataframe[feature].median()
            before_count = dataframe[feature].isnull().sum()
            dataframe[feature].fillna(median_value, inplace=True)
            print(f"{feature}: заполнено {before_count} пропусков медианой ({median_value})")
    
    # Для остальных категориальных признаков используем моду
    categorical_features = dataframe.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        if dataframe[feature].isnull().sum() > 0:
            mode_value = dataframe[feature].mode()[0]
            before_count = dataframe[feature].isnull().sum()
            dataframe[feature].fillna(mode_value, inplace=True)
            print(f"{feature}: заполнено {before_count} пропусков модой ('{mode_value}')")
    
    return dataframe

df_processed = handle_missing_values(df_processed)

# Проверяем результат
remaining_missing = df_processed.isnull().sum().sum()
print(f"\nОставшиеся пропуски: {remaining_missing}")

if remaining_missing == 0:
    print("✅ Все пропуски успешно обработаны!")

# 4.2. Альтернативные методы обработки пропусков

# Демонстрируем альтернативные методы для числовых признаков
def compare_imputation_methods(dataframe, feature, target='SalePrice'):
    """Сравнивает различные методы заполнения пропусков"""
    
    if feature not in dataframe.columns or dataframe[feature].isnull().sum() == 0:
        print(f"Признак {feature} не найден или не содержит пропусков")
        return
    
    # Создаем копии данных
    df_mean = dataframe.copy()
    df_median = dataframe.copy()
    df_knn = dataframe.copy()
    
    # Метод 1: Среднее значение
    mean_imputer = SimpleImputer(strategy='mean')
    df_mean[feature] = mean_imputer.fit_transform(df_mean[[feature]])
    
    # Метод 2: Медиана
    median_imputer = SimpleImputer(strategy='median')
    df_median[feature] = median_imputer.fit_transform(df_median[[feature]])
    
    # Метод 3: KNN Imputer (более сложный метод)
    # Для демонстрации используем только числовые признаки
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn[numeric_cols] = knn_imputer.fit_transform(df_knn[numeric_cols])
    
    # Сравниваем корреляции с целевой переменной
    print(f"СРАВНЕНИЕ МЕТОДОВ ЗАПОЛНЕНИЯ ДЛЯ '{feature}':")
    print("-" * 50)
    print(f"Корреляция с {target}:")
    print(f"  Среднее: {df_mean[feature].corr(df_mean[target]):.3f}")
    print(f"  Медиана: {df_median[feature].corr(df_median[target]):.3f}")
    print(f"  KNN: {df_knn[feature].corr(df_knn[target]):.3f}")
    
    return df_mean, df_median, df_knn

# Демонстрируем на примере LotFrontage (если есть пропуски)
if 'LotFrontage' in df.columns and df['LotFrontage'].isnull().sum() > 0:
    compare_imputation_methods(df, 'LotFrontage')

# Шаг 5: Feature Engineering
# 5.1. Создание новых признаков

def create_new_features(dataframe):
    """Создает новые признаки из существующих"""
    
    print("СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ:")
    print("="*50)
    
    df_new = dataframe.copy()
    
    # 1. Возраст дома
    if 'YearBuilt' in df_new.columns:
        current_year = 2023  # Можно использовать datetime.now().year
        df_new['HouseAge'] = current_year - df_new['YearBuilt']
        print("✅ Создан признак 'HouseAge' (возраст дома)")
    
    # 2. Возраст с момента последнего ремонта
    if 'YearRemodAdd' in df_new.columns:
        df_new['YearsSinceRemod'] = current_year - df_new['YearRemodAdd']
        print("✅ Создан признак 'YearsSinceRemod' (лет с ремонта)")
    
    # 3. Общая площадь дома
    area_features = ['TotalBsmtSF', 'GrLivArea']
    if all(col in df_new.columns for col in area_features):
        df_new['TotalSF'] = df_new['TotalBsmtSF'] + df_new['GrLivArea']
        print("✅ Создан признак 'TotalSF' (общая площадь)")
    
    # 4. Общее количество ванных комнат
    bath_features = ['FullBath', 'BedroomAbvGr']
    if all(col in df_new.columns for col in bath_features):
        df_new['TotalBathrooms'] = df_new['FullBath'] + df_new.get('HalfBath', 0) * 0.5
        print("✅ Создан признак 'TotalBathrooms' (общее кол-во ванных)")
    
    # 5. Отношение площади гаража к общей площади
    if all(col in df_new.columns for col in ['GarageArea', 'GrLivArea']):
        df_new['GarageRatio'] = df_new['GarageArea'] / (df_new['GrLivArea'] + 1)  # +1 чтобы избежать деления на 0
        print("✅ Создан признак 'GarageRatio' (отношение площади гаража)")
    
    # 6. Категориальные признаки на основе числовых
    if 'OverallQual' in df_new.columns:
        df_new['QualityCategory'] = pd.cut(df_new['OverallQual'], bins=[0, 4, 7, 10], labels=['Low', 'Medium', 'High'])
        print("✅ Создан признак 'QualityCategory' (категория качества)")
    
    # 7. Признак "премиум дом"
    if 'OverallQual' in df_new.columns and 'GrLivArea' in df_new.columns:
        df_new['IsPremium'] = ((df_new['OverallQual'] >= 8) & (df_new['GrLivArea'] > df_new['GrLivArea'].quantile(0.75))).astype(int)
        print("✅ Создан признак 'IsPremium' (премиум дом)")
    
    print(f"\nВсего создано новых признаков: {df_new.shape[1] - dataframe.shape[1]}")
    return df_new

df_processed = create_new_features(df_processed)

# Анализируем корреляции новых признаков
new_features = ['HouseAge', 'YearsSinceRemod', 'TotalSF', 'TotalBathrooms', 'GarageRatio', 'IsPremium']
existing_new_features = [f for f in new_features if f in df_processed.columns]

if existing_new_features:
    print(f"\nКорреляции новых признаков с SalePrice:")
    for feature in existing_new_features:
        if df_processed[feature].dtype in ['int64', 'float64']:
            corr = df_processed[feature].corr(df_processed['SalePrice'])
            print(f"  {feature}: {corr:.3f}")

# 5.2. Кодирование категориальных переменных

def encode_categorical_features(dataframe):
    """Кодирует категориальные признаки"""
    
    print("\nКОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")
    print("="*50)
    
    df_encoded = dataframe.copy()
    
    # Получаем категориальные признаки
    categorical_features = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Разделяем на порядковые и номинальные
    ordinal_features = {
        'OverallQual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'OverallCond': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'QualityCategory': ['Low', 'Medium', 'High']
    }
    
    # Кодируем порядковые признаки
    for feature, order in ordinal_features.items():
        if feature in df_encoded.columns:
            # Создаем маппинг
            mapping = {val: i for i, val in enumerate(order)}
            df_encoded[f'{feature}_encoded'] = df_encoded[feature].map(mapping)
            
            # Заполняем пропуски, если есть
            if df_encoded[f'{feature}_encoded'].isnull().sum() > 0:
                df_encoded[f'{feature}_encoded'].fillna(0, inplace=True)
            
            print(f"✅ {feature} закодирован как порядковый")
    
    # Для номинальных признаков используем One-Hot Encoding
    nominal_features = [f for f in categorical_features if f not in ordinal_features.keys()]
    
    # Ограничиваем количество категорий для One-Hot Encoding
    for feature in nominal_features:
        unique_count = df_encoded[feature].nunique()
        if unique_count <= 10:  # Только если категорий не слишком много
            dummies = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            print(f"✅ {feature} закодирован с помощью One-Hot Encoding ({unique_count} категорий)")
        else:
            print(f"⚠ {feature} пропущен (слишком много категорий: {unique_count})")
    
    # Удаляем исходные категориальные признаки
    df_encoded = df_encoded.select_dtypes(exclude=['object', 'category'])
    
    print(f"\nИтоговое количество признаков: {df_encoded.shape[1]}")
    return df_encoded

df_processed = encode_categorical_features(df_processed)

# 5.3. Масштабирование признаков

def scale_features(dataframe, target_column='SalePrice'):
    """Масштабирует признаки для улучшения работы алгоритмов"""
    
    print("\nМАСШТАБИРОВАНИЕ ПРИЗНАКОВ:")
    print("="*50)
    
    # Разделяем на признаки и целевую переменную
    if target_column in dataframe.columns:
        X = dataframe.drop([target_column, 'Id'], axis=1, errors='ignore')
        y = dataframe[target_column]
    else:
        X = dataframe.drop(['Id'], axis=1, errors='ignore')
        y = None
    
    # Применяем логарифмическое преобразование к целевой переменной
    if y is not None:
        # Проверяем скошенность
        skewness = skew(y)
        if abs(skewness) > 0.5:
            y_transformed = np.log1p(y)  # log1p = log(1 + x), безопасно для нулевых значений
            print(f"✅ Целевая переменная преобразована логарифмически (skewness: {skewness:.3f} -> {skew(y_transformed):.3f})")
        else:
            y_transformed = y
            print(f"✅ Целевая переменная не требует преобразования (skewness: {skewness:.3f})")
    
    # Находим сильно скошенные числовые признаки
    numeric_features = X.select_dtypes(include=[np.number]).columns
    skewed_features = []
    
    for feature in numeric_features:
        if X[feature].nunique() > 10:  # Только для непрерывных признаков
            feature_skewness = skew(X[feature])
            if abs(feature_skewness) > 0.75:
                skewed_features.append(feature)
    
    print(f"Найдено {len(skewed_features)} сильно скошенных признаков")
    
    # Применяем логарифмическое преобразование к скошенным признакам
    X_transformed = X.copy()
    for feature in skewed_features:
        X_transformed[feature] = np.log1p(X_transformed[feature])
        print(f"  ✅ {feature} преобразован логарифмически")
    
    return X_transformed, y_transformed if y is not None else None, skewed_features

X_processed, y_processed, skewed_features = scale_features(df_processed)

print(f"\nИтоговые размеры данных:")
print(f"Признаки (X): {X_processed.shape}")
if y_processed is not None:
    print(f"Целевая переменная (y): {y_processed.shape}")

# Шаг 6: Построение и сравнение моделей
# 6.1. Подготовка данных для моделирования

# === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Гарантированное удаление любых NaN ===
print("\nПРОВЕРКА И ОЧИСТКА ДАННЫХ ОТ NaN:")
print("="*50)

# 1. Оставляем только числовые столбцы (на случай, если категориальные проскочили)
X_processed = X_processed.select_dtypes(include=[np.number])

# 2. Заполняем любые оставшиеся пропуски медианой по каждому столбцу
for col in X_processed.columns:
    if X_processed[col].isnull().sum() > 0:
        X_processed[col] = X_processed[col].fillna(X_processed[col].median())

# 3. Финальная проверка
nan_count = X_processed.isnull().sum().sum()
if nan_count == 0:
    print("✅ Все пропуски (NaN) успешно устранены. Данные готовы к обучению.")
else:
    print(f"⚠ ВНИМАНИЕ: Осталось {nan_count} пропусков. Применяем глобальное заполнение нулями.")
    X_processed = X_processed.fillna(0)
# ====================================================================


# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed,
    test_size=0.2,
    random_state=42
)

print("ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ:")
print("="*50)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Количество признаков: {X_train.shape[1]}")

# Дополнительное масштабирование для регуляризованных моделей
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Данные масштабированы для регуляризованных моделей")

# 6.2. Обучение различных моделей

def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Обучает и оценивает различные модели регрессии"""
    
    models = {}
    results = {}
    
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ:")
    print("="*50)
    
    # 1. Обычная линейная регрессия
    
    print("1. Обычная линейная регрессия...")
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    models['Linear'] = lr
    results['Linear'] = {
        'predictions': lr_pred,
        'mae': mean_absolute_error(y_test, lr_pred),
        'mse': mean_squared_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred)
    }
    print(f"  R² = {results['Linear']['r2']:.4f}")
    
    # 2. Ridge регрессия с подбором параметров
    
    print("2. Ridge регрессия с подбором параметров...")
    
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
    ridge_grid.fit(X_train_scaled, y_train)
    
    best_ridge = ridge_grid.best_estimator_
    ridge_pred = best_ridge.predict(X_test_scaled)
    
    models['Ridge'] = best_ridge
    results['Ridge'] = {
        'predictions': ridge_pred,
        'mae': mean_absolute_error(y_test, ridge_pred),
        'mse': mean_squared_error(y_test, ridge_pred),
        'r2': r2_score(y_test, ridge_pred),
        'best_alpha': ridge_grid.best_params_['alpha']
    }
    print(f"  Лучший alpha = {results['Ridge']['best_alpha']}")
    print(f"  R² = {results['Ridge']['r2']:.4f}")
    
    # 3. Lasso регрессия с подбором параметров
    
    print("3. Lasso регрессия с подбором параметров...")
    
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring='r2')
    lasso_grid.fit(X_train_scaled, y_train)
    
    best_lasso = lasso_grid.best_estimator_
    lasso_pred = best_lasso.predict(X_test_scaled)
    
    models['Lasso'] = best_lasso
    results['Lasso'] = {
        'predictions': lasso_pred,
        'mae': mean_absolute_error(y_test, lasso_pred),
        'mse': mean_squared_error(y_test, lasso_pred),
        'r2': r2_score(y_test, lasso_pred),
        'best_alpha': lasso_grid.best_params_['alpha'],
        'n_features_used': np.sum(best_lasso.coef_ != 0)
    }
    
    print(f"  Лучший alpha = {results['Lasso']['best_alpha']}")
    print(f"  Использовано признаков = {results['Lasso']['n_features_used']}/{len(best_lasso.coef_)}")
    print(f"  R² = {results['Lasso']['r2']:.4f}")
    
    return models, results

models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

# 6.3. Сравнение результатов

# Создаем таблицу сравнения результатов
comparison_df = pd.DataFrame({
    'Модель': list(results.keys()),
    'MAE': [results[model]['mae'] for model in results.keys()],
    'MSE': [results[model]['mse'] for model in results.keys()],
    'R²': [results[model]['r2'] for model in results.keys()]
}).sort_values('R²', ascending=False)

print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ МОДЕЛЕЙ:")
print("="*50)
print(comparison_df.round(4))

# Визуализация сравнения
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Сравнение R²
axes[0, 0].bar(comparison_df['Модель'], comparison_df['R²'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 0].set_title('Сравнение R² по моделям')
axes[0, 0].set_ylabel('R²')
for i, v in enumerate(comparison_df['R²']):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# График 2: Сравнение MAE
axes[0, 1].bar(comparison_df['Модель'], comparison_df['MAE'], color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 1].set_title('Сравнение MAE по моделям')
axes[0, 1].set_ylabel('MAE')

# График 3: Реальные vs предсказанные для лучшей модели
best_model = comparison_df.iloc[0]['Модель']
best_predictions = results[best_model]['predictions']

axes[1, 0].scatter(y_test, best_predictions, alpha=0.6)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Реальные значения')
axes[1, 0].set_ylabel('Предсказанные значения')
axes[1, 0].set_title(f'Реальные vs Предсказанные ({best_model})')

# График 4: Остатки для лучшей модели
residuals = y_test - best_predictions
axes[1, 1].scatter(best_predictions, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Предсказанные значения')
axes[1, 1].set_ylabel('Остатки')
axes[1, 1].set_title(f'График остатков ({best_model})')

plt.tight_layout()
plt.show()

print(f"\n✅ Лучшая модель: {best_model} с R² = {comparison_df.iloc[0]['R²']:.4f}")

# Шаг 7: Анализ важности признаков
# 7.1. Анализ коэффициентов

def analyze_feature_importance(models, feature_names):
    """Анализирует важность признаков в различных моделях"""
    
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ:")
    print("="*50)
    
    # Создаем DataFrame с коэффициентами всех моделей
    importance_df = pd.DataFrame({'Feature': feature_names})
    
    for model_name, model in models.items():
        if hasattr(model, 'coef_'):
            importance_df[f'{model_name}_coef'] = model.coef_
            importance_df[f'{model_name}_abs'] = np.abs(model.coef_)
    
    # Сортируем по важности в лучшей модели
    best_model = comparison_df.iloc[0]['Модель']
    if f'{best_model}_abs' in importance_df.columns:
        importance_df = importance_df.sort_values(f'{best_model}_abs', ascending=False)
    
    # Показываем топ-15 самых важных признаков
    top_features = importance_df.head(15)
    print(f"Топ-15 самых важных признаков (по модели {best_model}):")
    print(top_features[['Feature', f'{best_model}_coef', f'{best_model}_abs']].round(4))
    
    # Визуализация важности признаков
    plt.figure(figsize=(12, 8))
    
    # График важности для лучшей модели
    plt.subplot(1, 2, 1)
    top_10 = importance_df.head(10)
    plt.barh(range(len(top_10)), top_10[f'{best_model}_abs'], color='skyblue')
    plt.yticks(range(len(top_10)), top_10['Feature'])
    plt.xlabel('Абсолютное значение коэффициента')
    plt.title(f'Топ-10 важных признаков ({best_model})')
    plt.gca().invert_yaxis()
    
    # Сравнение коэффициентов между моделями
    plt.subplot(1, 2, 2)
    if 'Ridge_coef' in importance_df.columns and 'Lasso_coef' in importance_df.columns:
        plt.scatter(importance_df['Ridge_coef'], importance_df['Lasso_coef'], alpha=0.6)
        plt.xlabel('Ridge коэффициенты')
        plt.ylabel('Lasso коэффициенты')
        plt.title('Сравнение коэффициентов Ridge vs Lasso')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

importance_df = analyze_feature_importance(models, X_processed.columns.tolist())

# 7.2. Анализ отобранных признаков Lasso

# Анализируем, какие признаки отобрала Lasso регрессия
if 'Lasso' in models:
    lasso_model = models['Lasso']
    selected_features = X_processed.columns[lasso_model.coef_ != 0].tolist()
    rejected_features = X_processed.columns[lasso_model.coef_ == 0].tolist()
    
    print("АНАЛИЗ ОТБОРА ПРИЗНАКОВ LASSO:")
    print("="*50)
    print(f"Всего признаков: {len(X_processed.columns)}")
    print(f"Отобрано: {len(selected_features)}")
    print(f"Отклонено: {len(rejected_features)}")
    print(f"Процент отобранных: {len(selected_features)/len(X_processed.columns)*100:.1f}%")
    
    print(f"\nОтобранные признаки:")
    for feature in selected_features[:10]:  # Показываем первые 10
        coef = lasso_model.coef_[X_processed.columns.get_loc(feature)]
        print(f"  {feature}: {coef:.4f}")
    
    if len(selected_features) > 10:
        print(f"  ... и еще {len(selected_features) - 10} признаков")
    
    print(f"\nПримеры отклоненных признаков:")
    for feature in rejected_features[:5]:  # Показываем первые 5
        print(f"  {feature}")

# Шаг 8: Практическое применение и выводы
# 8.1. Предсказание для новых домов

def predict_house_price(models, scaler, feature_names, house_features):
    """Предсказывает цену дома для новых данных"""
    
    # Создаем DataFrame с признаками нового дома
    new_house_df = pd.DataFrame([house_features], columns=feature_names)
    
    # Масштабируем данные
    new_house_scaled = scaler.transform(new_house_df)
    
    print("ПРЕДСКАЗАНИЕ ЦЕН ДЛЯ НОВЫХ ДОМОВ:")
    print("="*50)
    
    # Делаем предсказания всеми моделями
    predictions = {}
    for model_name, model in models.items():
        if model_name == 'Linear':
            pred = model.predict(new_house_df)[0]
        else:
            pred = model.predict(new_house_scaled)[0]
        
        # Преобразуем обратно из логарифмической шкалы
        pred_price = np.expm1(pred)  # expm1 = exp(x) - 1, обратная к log1p
        predictions[model_name] = pred_price
        
        print(f"{model_name}: ${pred_price:,.2f}")
    
    return predictions

# Создаем пример нового дома для предсказания
# Используем средние значения для большинства признаков
example_house = X_processed.mean().values

print("Предсказание для дома со средними характеристиками:")
predictions = predict_house_price(models, scaler, X_processed.columns.tolist(), example_house)

# Сравниваем со средней ценой в датасете
actual_mean_price = np.expm1(y_processed.mean())
print(f"\nСредняя цена в датасете: ${actual_mean_price:,.2f}")
print(f"Разброс предсказаний: ${min(predictions.values()):,.2f} - ${max(predictions.values()):,.2f}")

# 8.2. Основные выводы

print("ОСНОВНЫЕ ВЫВОДЫ ИЗ АНАЛИЗА:")
print("="*60)

print("1. КАЧЕСТВО ДАННЫХ:")
print(f"  • Исходный датасет: {df.shape[0]} домов, {df.shape[1]} признаков")
print(f"  • Пропущенные значения: успешно обработаны")
print(f"  • Итоговое количество признаков: {X_processed.shape[1]}")

print("\n2. FEATURE ENGINEERING:")
print("  • Созданы новые признаки: возраст дома, общая площадь, и др.")
print("  • Применено логарифмическое преобразование к скошенным признакам")
print("  • Категориальные признаки закодированы")

print("\n3. СРАВНЕНИЕ МОДЕЛЕЙ:")
best_model = comparison_df.iloc[0]['Модель']
best_r2 = comparison_df.iloc[0]['R²']
print(f"  • Лучшая модель: {best_model}")
print(f"  • Лучший R²: {best_r2:.4f}")
print(f"  • Модель объясняет {best_r2*100:.1f}% вариации цен")

if 'Lasso' in results:
    lasso_features = results['Lasso']['n_features_used']
    total_features = len(X_processed.columns)
    print(f"  • Lasso отобрала {lasso_features} из {total_features} признаков")

print("\n4. ВАЖНЫЕ ФАКТОРЫ ЦЕНЫ:")
if len(importance_df) > 0:
    top_3_features = importance_df.head(3)['Feature'].tolist()
    print("  • Топ-3 самых важных признака:")
    for i, feature in enumerate(top_3_features, 1):
        print(f"    {i}. {feature}")

print("\n5. РЕКОМЕНДАЦИИ:")
print("  • Модель готова для практического использования")
print("  • Рекомендуется регулярно переобучать на новых данных")
print("  • Можно добавить внешние данные (экономические показатели, криминальная статистика)")
print("  • Стоит исследовать нелинейные модели для дальнейшего улучшения")


# Задания для самостоятельной работы

# Задание 1. Дополнительные признаки
def create_additional_features(dataframe):
    df_new = dataframe.copy()
    
    # PricePerSqFt
    if 'SalePrice' in df_new.columns and 'GrLivArea' in df_new.columns:
        df_new['PricePerSqFt'] = df_new['SalePrice'] / (df_new['GrLivArea'] + 1)
    
    # AgeCategory
    if 'HouseAge' in df_new.columns:
        df_new['AgeCategory'] = pd.cut(df_new['HouseAge'], bins=[0, 10, 30, 100], labels=['New', 'Medium', 'Old'])
    
    # HasGarage
    if 'GarageArea' in df_new.columns:
        df_new['HasGarage'] = (df_new['GarageArea'] > 0).astype(int)
    
    return df_new

# Задание 2. Улучшение обработки пропусков

def compare_imputation_methods(dataframe, feature, target='SalePrice'):
    """Сравнивает различные методы заполнения пропусков (из методички)"""
    if feature not in dataframe.columns or dataframe[feature].isnull().sum() == 0:
        print(f"Признак {feature} не найден или не содержит пропусков")
        return
    
    # Создаем копии данных
    df_mean = dataframe.copy()
    df_median = dataframe.copy()
    df_knn = dataframe.copy()
    
    # Метод 1: Среднее значение
    mean_imputer = SimpleImputer(strategy='mean')
    df_mean[feature] = mean_imputer.fit_transform(df_mean[[feature]])
    
    # Метод 2: Медиана
    median_imputer = SimpleImputer(strategy='median')
    df_median[feature] = median_imputer.fit_transform(df_median[[feature]])
    
    # Метод 3: KNN Imputer
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if target in numeric_cols:
        numeric_cols.remove(target)
        
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn[numeric_cols] = knn_imputer.fit_transform(df_knn[numeric_cols])
    
    # Сравниваем корреляции с целевой переменной
    print(f"СРАВНЕНИЕ МЕТОДОВ ЗАПОЛНЕНИЯ ДЛЯ '{feature}':")
    print("-" * 50)
    print(f"Корреляция с {target}:")
    print(f"  Среднее : {df_mean[feature].corr(df_mean[target]):.3f}")
    print(f"  Медиана : {df_median[feature].corr(df_median[target]):.3f}")
    print(f"  KNN     : {df_knn[feature].corr(df_knn[target]):.3f}")

# Демонстрируем на примере LotFrontage (классический признак с пропусками в Ames)
# Берем исходный df, так как в df_processed пропуски уже заполнены
if 'LotFrontage' in df.columns and df['LotFrontage'].isnull().sum() > 0:
    compare_imputation_methods(df, 'LotFrontage')
else:
    print("Признак LotFrontage не найден или не имеет пропусков в исходном df.")


# Задание 3. Анализ остатков

# Проходим по всем обученным моделям
for model_name, model in models.items():
    print(f"\n--- Анализ остатков для модели: {model_name} ---")
    
    # Делаем предсказания (учитываем, что Ridge и Lasso обучались на масштабированных данных)
    if model_name == 'Linear':
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test_scaled)
    
    # Вычисляем остатки
    residuals = y_test - preds
    
    # Строим графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Остатки vs Предсказанные значения
    axes[0].scatter(preds, residuals, alpha=0.6, color='green')
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Предсказанные значения (log SalePrice)')
    axes[0].set_ylabel('Остатки')
    axes[0].set_title(f'Остатки vs Предсказанные ({model_name})')
    
    # График 2: Q-Q plot для проверки нормальности
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q plot остатков ({model_name})')
    
    plt.tight_layout()
    plt.show()

# Задание 4. Кросс-валидация

def cross_validate_models(models, X, y, cv=10):
    """Применяет кросс-валидацию ко всем моделям и выводит результаты"""
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ:")
    print("="*40)
    
    for model_name, model in models.items():
        if model_name == 'Linear':
            # Для обычной линейной регрессии используем исходные данные
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        else:
            # Для регуляризованных моделей (Ridge, Lasso) данные нужно масштабировать
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        
        print(f"Модель: {model_name}")
        print(f"  Среднее R²:          {scores.mean():.4f}")
        print(f"  Стд. отклонение:     {scores.std():.4f}")
        print(f"  95% доверительный интервал: [{scores.mean() - 2*scores.std():.4f}, {scores.mean() + 2*scores.std():.4f}]")
        print()

# Запуск кросс-валидации
cross_validate_models(models, X_processed, y_processed, cv=10)