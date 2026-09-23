# Основные библиотеки для работы с данными
import pandas as pd
import numpy as np
# Библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns
# Библиотеки для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# Настройки для красивых графиков
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
# Отключаем предупреждения
import warnings
warnings.filterwarnings('ignore')
print("Все библиотеки успешно импортированы!")

# Загружаем датасет
# Если у вас есть файл insurance.csv, используйте эту строку:
# df = pd.read_csv('insurance.csv')
# Для демонстрации создадим небольшой пример данных
# В реальной работе вы будете загружать полный датасет
sample_data = {
'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60],
'sex': ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male',
'female'],
'bmi': [27.9, 33.77, 33.0, 22.705, 28.88, 25.74, 33.44, 27.74, 29.83, 25.84],
'children': [0, 1, 3, 0, 0, 0, 1, 3, 2, 0],
'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest',
'southeast', 'southeast', 'northwest', 'northeast', 'northwest'],
'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552,
3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692]
}
df = pd.DataFrame(sample_data)
print("Данные успешно загружены!")
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")


# Смотрим на первые 5 строк
print("Первые 5 строк данных:")
print(df.head())
print("\n" + "="*50)
# Смотрим на последние 5 строк
print("Последние 5 строк данных:")
print(df.tail())
print("\n" + "="*50)
# Получаем общую информацию о датасете
print("Общая информация о данных:")
print(df.info())

# Проверяем пропущенные значения
missing_values = df.isnull().sum()
print("Пропущенные значения в каждом столбце:")
print(missing_values)
if missing_values.sum() == 0:
	print("\nОтлично! Пропущенных значений нет.")
else:
	print(f"\nНайдено {missing_values.sum()} пропущенных значений.")


# Описательная статистика для числовых столбцов
print("Описательная статистика для числовых признаков:")
print(df.describe())
print("\n" + "="*50)
# Информация о категориальных столбцах
categorical_columns = ['sex', 'smoker', 'region']
print("Информация о категориальных признаках:")
for col in categorical_columns:
	print(f"\n{col}:")
	print(df[col].value_counts())


# Создаем фигуру с несколькими подграфиками
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# Гистограмма распределения стоимости страховки
axes[0, 0].hist(df['charges'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Распределение стоимости страховки')
axes[0, 0].set_xlabel('Стоимость (charges)')
axes[0, 0].set_ylabel('Количество клиентов')
# Ящик с усами для charges
axes[0, 1].boxplot(df['charges'])
axes[0, 1].set_title('Ящик с усами для стоимости страховки')
axes[0, 1].set_ylabel('Стоимость (charges)')
# Гистограмма с кривой плотности
sns.histplot(df['charges'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Распределение с кривой плотности')
axes[1, 0].set_xlabel('Стоимость (charges)')

# Статистика
axes[1, 1].text(0.1, 0.8, f'Среднее: ${df["charges"].mean():.2f}', fontsize=12,
transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.7, f'Медиана: ${df["charges"].median():.2f}', fontsize=12,
transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.6, f'Стд. отклонение: ${df["charges"].std():.2f}', fontsize=12,
transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.5, f'Минимум: ${df["charges"].min():.2f}', fontsize=12,
transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.4, f'Максимум: ${df["charges"].max():.2f}', fontsize=12,
transform=axes[1, 1].transAxes)
axes[1, 1].set_title('Основная статистика')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()

# Выводы о распределении
print("Анализ целевой переменной (charges):")
print(f"• Среднее значение: ${df['charges'].mean():.2f}")
print(f"• Медиана: ${df['charges'].median():.2f}")
print(f"• Разброс данных: от ${df['charges'].min():.2f} до ${df['charges'].max():.2f}")
if df['charges'].mean() > df['charges'].median():
	print("• Распределение смещено вправо (есть клиенты с очень высокой стоимостью)")
else:
	print("• Распределение симметричное")

# Анализируем связь числовых признаков с целевой переменной
numeric_features = ['age', 'bmi', 'children']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel() # Делаем массив одномерным для удобства
for i, feature in enumerate(numeric_features):
# Диаграмма рассеяния
	axes[i].scatter(df[feature], df['charges'], alpha=0.6, color='coral')
	axes[i].set_xlabel(feature)
	axes[i].set_ylabel('charges')
	axes[i].set_title(f'Зависимость charges от {feature}')
	# Добавляем линию тренда
	z = np.polyfit(df[feature], df['charges'], 1)
	p = np.poly1d(z)
	axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)

# Корреляционная матрица
numeric_df = df[numeric_features + ['charges']]
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[3])
axes[3].set_title('Корреляционная матрица')
plt.tight_layout()
plt.show()

# Выводим корреляции с целевой переменной
print("Корреляция числовых признаков с charges:")
correlations = df[numeric_features +
['charges']].corr()['charges'].sort_values(ascending=False)
for feature, corr in correlations.items():
	if feature != 'charges':
		print(f"• {feature}: {corr:.3f}")
		if abs(corr) > 0.5:
			print(" → Сильная связь")
		elif abs(corr) > 0.3:
			print(" → Умеренная связь")
		else:
			print(" → Слабая связь")

# Анализируем категориальные признаки
categorical_features = ['sex', 'smoker', 'region']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(categorical_features):
# Ящик с усами для каждой категории
	df.boxplot(column='charges', by=feature, ax=axes[i])
	axes[i].set_title(f'Распределение charges по {feature}')
	axes[i].set_xlabel(feature)
	axes[i].set_ylabel('charges')

# Столбчатая диаграмма средних значений
mean_charges = df.groupby('smoker')['charges'].mean()
axes[3].bar(mean_charges.index, mean_charges.values, color=['lightcoral', 'lightblue'])
axes[3].set_title('Средняя стоимость страховки: курящие vs некурящие')
axes[3].set_xlabel('Статус курения')
axes[3].set_ylabel('Средняя стоимость')
# Добавляем значения на столбцы
for i, v in enumerate(mean_charges.values):
	axes[3].text(i, v + 500, f'${v:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
# Статистический анализ категориальных признаков
print("Анализ категориальных признаков:")
for feature in categorical_features:
	print(f"\n{feature.upper()}:")
	group_stats = df.groupby(feature)['charges'].agg(['mean', 'median', 'std', 'count'])
	print(group_stats)


# Ищем выбросы в данных
def find_outliers_iqr(data, column):
	"""Находит выбросы методом межквартильного размаха (IQR)"""
	Q1 = data[column].quantile(0.25)
	Q3 = data[column].quantile(0.75)
	IQR = Q3 - Q1
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
	return outliers, lower_bound, upper_bound

# Анализируем выбросы для числовых признаков
print("Анализ выбросов:")
for feature in ['age', 'bmi', 'children', 'charges']:
	outliers, lower, upper = find_outliers_iqr(df, feature)
	print(f"\n{feature}:")
	print(f" Границы: [{lower:.2f}, {upper:.2f}]")
	print(f" Количество выбросов: {len(outliers)}")
	if len(outliers) > 0:
		print(f" Процент выбросов: {len(outliers)/len(df)*100:.1f}%")

# Создаем копию данных для обработки
df_processed = df.copy()
print("Исходные категориальные признаки:")
print(df_processed[['sex', 'smoker', 'region']].head())
# Метод 1: Label Encoding для бинарных признаков
label_encoder = LabelEncoder()
# Кодируем пол (male/female)
df_processed['sex_encoded'] = label_encoder.fit_transform(df_processed['sex'])
print(f"\nКодирование пола:")
print(f"male → {label_encoder.transform(['male'])[0]}")
print(f"female → {label_encoder.transform(['female'])[0]}")
# Кодируем статус курения (yes/no)
df_processed['smoker_encoded'] = label_encoder.fit_transform(df_processed['smoker'])
print(f"\nКодирование курения:")
print(f"no → {label_encoder.transform(['no'])[0]}")
print(f"yes → {label_encoder.transform(['yes'])[0]}")
# Метод 2: One-Hot Encoding для региона (больше 2 категорий)
region_dummies = pd.get_dummies(df_processed['region'], prefix='region')
df_processed = pd.concat([df_processed, region_dummies], axis=1)
print(f"\nOne-Hot Encoding для региона:")
print(region_dummies.head())
# Удаляем исходные категориальные столбцы
df_processed = df_processed.drop(['sex', 'smoker', 'region'], axis=1)
print(f"\nИтоговые столбцы после кодирования:")
print(df_processed.columns.tolist())

# Проверяем финальный датасет
print("Финальный датасет для модели:")
print(df_processed.head())
print(f"\nРазмер: {df_processed.shape}")
print(f"Типы данных:")
print(df_processed.dtypes)
# Проверяем корреляции после кодирования
plt.figure(figsize=(12, 10))
correlation_matrix = df_processed.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица после обработки данных')
plt.tight_layout()
plt.show()
# Выводим корреляции с целевой переменной
print("\nКорреляция всех признаков с charges:")
target_correlations = correlation_matrix['charges'].sort_values(ascending=False)
for feature, corr in target_correlations.items():
	if feature != 'charges':
		print(f"• {feature}: {corr:.3f}")

# Определяем признаки (X) и целевую переменную (y)
X = df_processed.drop('charges', axis=1)
y = df_processed['charges']
print("Признаки (X):")
print(X.columns.tolist())
print(f"Размер X: {X.shape}")
print(f"\nЦелевая переменная (y):")
print(f"Размер y: {y.shape}")
print(f"Тип: {type(y)}")
# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2, # 20% данных для тестирования
random_state=42 # Для воспроизводимости результатов
)
print(f"\nРазделение данных:")
print(f"Обучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")
print(f"Процент тестовых данных: {X_test.shape[0]/len(df)*100:.1f}%")

# Создаем модель линейной регрессии
model = LinearRegression()
print("Обучаем модель...")
# Обучаем модель на тренировочных данных
model.fit(X_train, y_train)
print("� Модель успешно обучена!")
# Смотрим на коэффициенты модели
print(f"\nСвободный член (intercept): ${model.intercept_:.2f}")
print(f"\nКоэффициенты модели:")
coefficients_df = pd.DataFrame({
'Признак': X.columns,
'Коэффициент': model.coef_,
'Абсолютное значение': np.abs(model.coef_)
}).sort_values('Абсолютное значение', ascending=False)
print(coefficients_df)

# Интерпретируем коэффициенты
print("Интерпретация коэффициентов:")
print("(Как изменится стоимость страховки при изменении признака на 1 единицу)")
for feature, coef in zip(X.columns, model.coef_):
	if abs(coef) > 100: # Показываем только значимые коэффициенты
		if coef > 0:
			print(f"• {feature}: +${coef:.2f} (увеличивает стоимость)")
		else:
			print(f"• {feature}: ${coef:.2f} (уменьшает стоимость)")
# Находим самые важные признаки
most_important = coefficients_df.head(3)
print(f"\nТоп-3 самых важных признака:")
for _, row in most_important.iterrows():
	print(f"• {row['Признак']}: {row['Коэффициент']:.2f}")

# Делаем предсказания на тестовой выборке
y_pred = model.predict(X_test)
print("Предсказания модели:")
print("Реальные vs Предсказанные значения (первые 10):")
comparison_df = pd.DataFrame({
'Реальные': y_test.values,
'Предсказанные': y_pred,
'Разность': y_test.values - y_pred,
'Абс. ошибка': np.abs(y_test.values - y_pred)
})
print(comparison_df.head(10))

# Рассчитываем основные метрики
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Метрики качества модели:")
print("="*40)

print(f"MAE (Средняя абсолютная ошибка): ${mae:.2f}")
print(f"MSE (Средняя квадратичная ошибка): ${mse:.2f}")
print(f"RMSE (Корень из MSE): ${rmse:.2f}")
print(f"R² (Коэффициент детерминации): {r2:.3f}")
print("\nИнтерпретация метрик:")
print(f"• В среднем модель ошибается на ${mae:.2f}")
print(f"• Модель объясняет {r2*100:.1f}% вариации в данных")
if r2 > 0.8:
	print("• Отличное качество модели!")
elif r2 > 0.6:
	print("• Хорошее качество модели")
elif r2 > 0.4:
	print("• Удовлетворительное качество модели")
else:
	print("• Модель требует улучшения")


# Создаем графики для оценки качества
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# График 1: Реальные vs Предсказанные значения
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Реальные значения')
axes[0, 0].set_ylabel('Предсказанные значения')
axes[0, 0].set_title('Реальные vs Предсказанные значения')
# График 2: Остатки (residuals)
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Предсказанные значения')
axes[0, 1].set_ylabel('Остатки')
axes[0, 1].set_title('График остатков')
# График 3: Распределение остатков
axes[1, 0].hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Остатки')
axes[1, 0].set_ylabel('Частота')
axes[1, 0].set_title('Распределение остатков')
# График 4: Q-Q plot для проверки нормальности остатков
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q график остатков')
plt.tight_layout()
plt.show()

# Создаем профили новых клиентов для предсказания
new_clients = pd.DataFrame({
'age': [25, 45, 35],
'bmi': [22.5, 30.0, 25.0],
'children': [0, 2, 1],
'sex_encoded': [0, 1, 0], # 0 = female, 1 = male
'smoker_encoded': [0, 1, 0], # 0 = no, 1 = yes
'region_northeast': [1, 0, 0],
'region_northwest': [0, 1, 0],
'region_southeast': [0, 0, 1],
'region_southwest': [0, 0, 0]
})

print("Профили новых клиентов:")
print(new_clients)
# Делаем предсказания
predictions = model.predict(new_clients)
print(f"\nПредсказания стоимости страховки:")
for i, pred in enumerate(predictions):
	print(f"Клиент {i+1}: ${pred:.2f}")

# Создаем понятное описание клиентов
client_descriptions = [
"25-летняя женщина, ИМТ 22.5, без детей, не курит, северо-восток",
"45-летний мужчина, ИМТ 30.0, 2 детей, курит, северо-запад",
"35-летняя женщина, ИМТ 25.0, 1 ребенок, не курит, юго-восток"
]

print(f"\nПодробные предсказания:")
for i, (desc, pred) in enumerate(zip(client_descriptions, predictions)):
	print(f"{i+1}. {desc}")
	print(f" Предсказанная стоимость: ${pred:.2f}")
	print()

# Создаем график важности признаков
feature_importance = pd.DataFrame({
'Признак': X.columns,
'Важность': np.abs(model.coef_)
}).sort_values('Важность', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Признак'], feature_importance['Важность'])
plt.xlabel('Абсолютное значение коэффициента')
plt.title('Важность признаков в модели')

plt.tight_layout()
plt.show()
print("Ранжирование признаков по важности:")
for i, (_, row) in enumerate(feature_importance.sort_values('Важность',
ascending=False).iterrows()):
	print(f"{i+1}. {row['Признак']}: {row['Важность']:.2f}")

print("ОСНОВНЫЕ ВЫВОДЫ ИЗ АНАЛИЗА:")
print("="*50)
# Анализ данных
print("1. АНАЛИЗ ДАННЫХ:")
print(f" • Датасет содержит {len(df)} записей без пропусков")
print(f" • Средняя стоимость страховки: ${df['charges'].mean():.2f}")
print(f" • Разброс стоимости: от ${df['charges'].min():.2f} до ${df['charges'].max():.2f}")

# Важные закономерности
print("\n2. ВАЖНЫЕ ЗАКОНОМЕРНОСТИ:")
smoker_effect = df.groupby('smoker')['charges'].mean()
if len(smoker_effect) > 1:
	difference = smoker_effect['yes'] - smoker_effect['no']
	print(f" • Курящие платят в среднем на ${difference:.2f} больше")

age_corr = df['age'].corr(df['charges'])
print(f" • Корреляция возраста со стоимостью: {age_corr:.3f}")
bmi_corr = df['bmi'].corr(df['charges'])
print(f" • Корреляция ИМТ со стоимостью: {bmi_corr:.3f}")
# Качество модели
print(f"\n3. КАЧЕСТВО МОДЕЛИ:")
print(f" • R² = {r2:.3f} (модель объясняет {r2*100:.1f}% вариации)")
print(f" • Средняя ошибка: ${mae:.2f}")
print(f" • Самый важный фактор: {feature_importance.iloc[-1]['Признак']}")

print("\nРЕКОМЕНДАЦИИ ДЛЯ СТРАХОВОЙ КОМПАНИИ:")
print("="*50)
print("1. ЦЕНООБРАЗОВАНИЕ:")
print(" • Основной фактор риска - курение")
print(" • Возраст и ИМТ также влияют на стоимость")
print(" • Количество детей имеет умеренное влияние")
print("\n2. УЛУЧШЕНИЕ МОДЕЛИ:")

print(" • Собрать больше данных о здоровье клиентов")
print(" • Добавить информацию о хронических заболеваниях")
print(" • Учесть семейную историю болезней")
print("\n3. БИЗНЕС-ПРИМЕНЕНИЕ:")
print(" • Использовать модель для автоматического расчета премий")
print(" • Выявлять клиентов с высоким риском")
print(" • Разрабатывать программы стимулирования здорового образа жизни")