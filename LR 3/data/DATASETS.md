# Данные Лабораторной работы 2: деревья решений и ансамбли

Все файлы в этой папке — локальные фиксированные копии для офлайн-запуска
ноутбуков. Их контрольные суммы позволяют проверить целостность поставки.

## Bank Marketing — часть «Деревья и ансамбли в sklearn по CRISP-DM» Лабораторной работы 2

- Файл: `bank-full.csv`.
- Источник: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%26).
- Цитирование: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*.
  DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306).
- Лицензия: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Состав: 45 211 наблюдений, 16 признаков и бинарная цель `y`; разделитель — `;`.
- Извлечён из официального вложенного архива `bank.zip`; SHA-256 итогового CSV:
  `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`.
- Важное ограничение: `duration` описывает уже завершённый звонок и не входит
  в основную модель, принимающую решение до звонка.

## Rice (Cammeo and Osmancik) — часть «Реализация CART с нуля» Лабораторной работы 2

- Файлы: `rice/raw/rice-cammeo-osmancik-uci.zip` и
  `rice/raw/Rice_Cammeo_Osmancik.arff`.
- Источник: [UCI Machine Learning Repository — Rice (Cammeo and
  Osmancik)](https://archive.ics.uci.edu/dataset/545/rice%2Bcammeo%2Band%2Bosmancik),
  DOI [10.24432/C5MW4Z](https://doi.org/10.24432/C5MW4Z).
- Лицензия: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Состав: 3 810 наблюдений, 7 числовых морфологических признаков и два класса
  (`Cammeo`, `Osmancik`); пропусков нет.
- Исходный ARFF сохранён без обработки. Он сгруппирован по классам, поэтому в
  ноутбуках строки явно перемешиваются до stratified train/validation/test split.
- SHA-256 архива: `fe94e42046b829de21b92b0ffb6a22774fde021328cae799faa802a14e8dbed9`;
  SHA-256 ARFF: `1af97883100c89de2ea2972f7a28d428f4f1c14711a61defc0b0569e9eb65665`.
  Полная карточка, схема и атрибуция находятся в
  [`rice/README.md`](rice/README.md).
