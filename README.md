# HDR Image Processing Project (Python Version)

## Описание проекта

Этот проект реализует систему для создания изображений с высоким динамическим диапазоном (HDR) из серии фотографий с разной экспозицией. Он использует Flask для создания веб-интерфейса, через который пользователи могут загружать изображения и просматривать результаты обработки HDR.

## Особенности

- Создание HDR-изображений из серии снимков с различной экспозицией.
- Веб-интерфейс для загрузки и обработки изображений.
- Тоновое отображение для преобразования HDR-изображений в формат, пригодный для отображения на стандартных дисплеях.

## Требования

Для запуска проекта необходимо установить следующие библиотеки:
- OpenCV
- NumPy
- Flask
- Matplotlib

## Установка и Запуск

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/barashevdamir/HDRDebvec-Python-.git
   ```
2. Перейдите в директорию проекта:
   ```bash
   cd HDRDebvec-Python-
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Запустите сервер Flask:
   ```bash
   python main.py
   ```
## Использование

После запуска веб-сервера выполните следующие шаги:

1. **Перейдите на главную страницу.** Откройте веб-интерфейс, введя адрес `http://127.0.0.1:5000/` в вашем браузере.

2. **Загрузите изображения.** Выберите и загрузите серию изображений с разной экспозицией через веб-форму.

3. **Введите времена экспозиции.** Для каждого загруженного изображения укажите соответствующее время экспозиции.

4. **Нажмите кнопку "Обработать".** Это инициирует процесс создания HDR-изображения из загруженных фотографий.

5. **Просмотрите и скачайте результаты.** После обработки вы можете просмотреть HDR-изображение и скачать его через веб-интерфейс.

## Структура проекта

```plaintext
.
├── main.py                  # Основной файл приложения Flask
├── requirements.txt         # Файл зависимостей Python
├── uploads/                 # Папка для загружаемых изображений
├── results/                 # Папка для результатов обработки
└── templates/               # HTML-шаблоны для веб-интерфейса
```

# Описание кода проекта HDR Image Processing

## Импорт необходимых библиотек
```python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from flask_session import Session
from werkzeug.utils import secure_filename

```
Этот код отвечает за импорт всех необходимых библиотек. Он включает библиотеки для работы с изображениями (OpenCV, matplotlib), математические операции (NumPy, math), обработку предупреждений, многопоточность, а также инструменты Flask для создания веб-приложения.

## Настройка Flask-приложения
```python
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

```
Здесь конфигурируется Flask-приложение. Устанавливаются пути для папок загрузки и результатов, а также настройки сессии. Создаются директории для хранения загруженных изображений и результатов обработки.

## Класс HDR для обработки HDR-изображений
```python
class HDR:

    def __init__(self, path, filenames, exposure_times):
        with ThreadPoolExecutor() as executor:
            self.images = list(executor.map(cv2.imread, filenames))

        self.times = np.array(exposure_times, dtype=np.float32)
        self.N = len(self.images)
        self.row = len(self.images[0])
        self.col = len(self.images[0][0])

        # Выравнивание исходных изображений
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(self.images, self.images)

        # Коэффициент веса для плавности кривых отклика
        self.l = 10
```
Определение класса HDR, который будет обрабатывать изображения для создания HDR-изображения.
Конструктор загружает изображения, устанавливает времена экспозиции и выравнивает их с помощью алгоритма MTB (Median Threshold Bitmap).
Класс HDR предназначен для создания изображений с высоким динамическим диапазоном (HDR) из серии фотографий с разной экспозицией.
Это достигается путем объединения информации о свете и тени из разных изображений для создания одного изображения с более широким диапазоном яркости.

## Методы класса HDR
```python
def weightingFunction(self):
  # Определение минимального и максимального значения пикселей, которые могут быть в изображении.
  Zmin = 0  # Минимальное значение пикселя (черный).
  Zmax = 255  # Максимальное значение пикселя (белый).
  Zmid = (Zmax + Zmin) // 2

  # Инициализация массива для весовой функции, который будет хранить вес каждого возможного значения пикселя.
  self.w = np.zeros((Zmax - Zmin + 1))

  # Заполнение весовой функции значениями.
  # Веса распределяются таким образом, чтобы значения пикселей, близкие к среднему (128), имели больший вес,
  # а значения, близкие к краям диапазона (0 или 255), имели меньший вес.
  for z in range(Zmin, Zmax + 1):
      if z <= Zmid:
          self.w[z] = z - Zmin + 1
      else:
          self.w[z] = Zmax - z + 1

```

Метод weightingFunction создает весовую функцию, которая придает больший вес хорошо экспонированным пикселям и меньший вес плохо экспонированным.
Это важно для минимизации шума и артефактов в конечном HDR-изображении.
```python
def samplingValues(self):
  # Определение количества образцов для выборки на основе количества изображений и диапазона значений пикселей.
  samples = math.ceil(255 * 2 / (self.N - 1)) * 2  # Количество образцов для выборки.
  pixels = self.row * self.col  # Общее количество пикселей в изображении.

  # Определение шага выборки для равномерного распределения выборки по всему изображению.
  step = int(np.floor(pixels / samples))  # Шаг выборки пикселей.
  self.indices = list(range(0, pixels, step))[:-1]  # Индексы выбранных пикселей.

  # Создание плоского (одномерного) представления изображений для упрощения доступа к пикселям.
  self.flattenImage = np.zeros((self.N, 3, pixels), dtype=np.uint8)  # Массив для плоских изображений.

  def flatten_channel(channel_index):
      for i in range(self.N):
          # Преобразование каждого цветового канала в одномерный массив.
          self.flattenImage[i, channel_index] = np.reshape(self.images[i][:, :, channel_index], (pixels,))

  # Использование многопоточности для ускорения процесса плоского представления изображений
  with ThreadPoolExecutor(max_workers=3) as executor:
      executor.map(flatten_channel, range(3))

  # Логарифмирование времен экспозиции для использования в математических расчетах.
  X = np.log(self.times)  # Логарифмические времена экспозиции.

  # Массивы для хранения значений пикселей в выбранных местах
  self.ZB = np.zeros((samples, self.N), dtype=np.uint8)
  self.ZG = np.zeros((samples, self.N), dtype=np.uint8)
  self.ZR = np.zeros((samples, self.N), dtype=np.uint8)

  # Получение выбранных значений пикселей
  for k in range(self.N):
      self.ZB[:, k] = self.flattenImage[k, 0][self.indices]
      self.ZG[:, k] = self.flattenImage[k, 1][self.indices]
      self.ZR[:, k] = self.flattenImage[k, 2][self.indices]

  ind = np.arange(0, samples)  # Индексы всех выборок.

  # Обновление массивов значений пикселей после исключения неподходящих выборок.
  self.ZB = self.ZB[ind]
  self.ZG = self.ZG[ind]
  self.ZR = self.ZR[ind]

  # Массивы для хранения логарифмических времен экспозиции
  r, c = self.ZG.shape[:2]
  self.Bij = np.tile(np.log(self.times), (self.row * self.col, 1))

```
Метод samplingValues выбирает пиксели из каждого изображения для определения функции отклика камеры.
Это ключевой шаг в процессе создания HDR, так как функция отклика камеры необходима для правильного объединения различных экспозиций.
```python
def CRFsolve(self, Z):
   
  n = 256

  s1, s2 = Z.shape
  U = np.zeros((s1 * s2 + n + 1, n + s1))
  V = np.zeros((U.shape[0], 1))

  # Здесь создаются матрица U и вектор V, которые будут использоваться для решения системы линейных уравнений.
  # Размеры этих матриц зависят от количества пикселей в изображениях (self.Z.shape) и количества возможных значений пикселей (256 для 8-битного изображения).

  # Включение уравнений для соответствия данным
  k = 0
  for i in range(s1):
      for j in range(s2):
          wij = self.w[Z[i, j]]
          U[k, Z[i, j]] = wij
          U[k, n + i] = -wij
          V[k] = wij * self.Bij[i, j]
          k += 1

  # Этот двойной цикл заполняет матрицу U и вектор V.
  # Для каждого пикселя в каждом изображении создается уравнение, которое связывает значение пикселя с соответствующим значением времени экспозиции.
  # Веса wij используются для уменьшения влияния пикселей с очень высокой или очень низкой освещенностью.

  # Фиксация кривой путем установки ее среднего значения в ноль
  U[k, 129] = 0
  k += 1

  # Включение уравнений для плавности
  for i in range(1, n - 2):
      U[k, i] = self.l * self.w[i + 1]
      U[k, i + 1] = -2 * self.l * self.w[i + 1]
      U[k, i + 2] = self.l * self.w[i + 1]
      k += 1
  # Здесь добавляются уравнения для гладкости функции отображения камеры.
  # Это помогает гарантировать, что функция будет изменяться плавно, что важно для точного восстановления освещенности.

  # Решение системы с использованием метода наименьших квадратов
  M = np.dot(np.linalg.pinv(U), V)
  CRF = M[0:n]
  logE = M[n: len(M)]

  return CRF, logE

```

Метод CRFsolve решает систему уравнений для получения функции отклика камеры и логарифмических значений освещенности.
Это математически сложный процесс, который требует решения большого количества уравнений для получения точной функции отклика.

Задан набор значений пикселей, наблюдаемых для нескольких пикселей на нескольких изображениях с разным временем экспозиции,
эта функция возвращает функцию отклика камеры g, а также логарифмические значения освещенности для наблюдаемых пикселей

Входные значения:
Z(i,j): пиксельные значения местоположений пикселей с номером i на изображении j
B(j): логарифмическая дельта t для изображения j
l: лямбда, константа, определяющая степень сглаживания
w(z): весовая функция значения пикселя z

Выходные значения:
g(z) : логарифмическая экспозиция, соответствующая значению пикселя z
lE(i): логарифмическая освещенность в местоположении пикселя i

```python
 def process(self):
   
     self.weightingFunction()
     self.samplingValues()

     with ThreadPoolExecutor() as executor:
         futures = [executor.submit(self.CRFsolve, self.ZR),
                    executor.submit(self.CRFsolve, self.ZG),
                    executor.submit(self.CRFsolve, self.ZB)]

         self.CRFB, self.lEB = futures[0].result()
         self.CRFG, self.lEG = futures[1].result()
         self.CRFR, self.lER = futures[2].result()
```

Вызывает предыдущие методы для построения функции отклика и подготовки данных для восстановления карты освещенности HDR.

## Наследование и расширение класса HDR в PostProcess

```python
class PostProcess(HDR):

    def __init__(self, HDR):
        super().__init__(path, filenames, exposure_times)
```

Класс PostProcess наследует HDR и добавляет методы для сохранения и тонового отображения HDR-изображения.
Тоновое отображение необходимо, потому что стандартные дисплеи не могут отображать HDR напрямую.

## Методы класса PostProcess

```python
 def save_hdr_image(self, filename='results/hdr_image.hdr'):

     cv2.imwrite(filename, self.imgf32)
```
Метод save_hdr_image сохраняет HDR-изображение в формате, который может быть использован другими программами и устройствами, поддерживающими HDR.

```python
 def recover_HDR_RadianceMap(self):
   
     m = np.zeros((self.flattenImage.shape[1:]))
     wsum = np.zeros(self.flattenImage.shape[1:])
     hdr = np.zeros(self.flattenImage.shape[1:])
     # Здесь создаются массивы m, wsum, и hdr для хранения промежуточных и конечных результатов.
     # lnDt - это логарифм времен экспозиции, используемых для съемки серии изображений.

     lnDt = np.log(self.times)

     for i in range(self.N):
         wij_B = self.w[self.flattenImage[i, 0]]
         wij_G = self.w[self.flattenImage[i, 1]]
         wij_R = self.w[self.flattenImage[i, 2]]

         wsum[0, :] += wij_B
         wsum[1, :] += wij_G
         wsum[2, :] += wij_R

         # В этом цикле для каждого изображения в серии вычисляются веса wij_B, wij_G, и wij_R для каждого канала (синего, зеленого и красного соответственно).
         # Эти веса используются для уменьшения влияния пикселей с очень высокой или очень низкой освещенностью.

         m0 = np.subtract(self.CRFB[self.flattenImage[i, 0]], lnDt[i])[:, 0]
         m1 = np.subtract(self.CRFG[self.flattenImage[i, 1]], lnDt[i])[:, 0]
         m2 = np.subtract(self.CRFR[self.flattenImage[i, 2]], lnDt[i])[:, 0]

         hdr[0] = hdr[0] + np.multiply(m0, wij_B)
         hdr[1] = hdr[1] + np.multiply(m1, wij_G)
         hdr[2] = hdr[2] + np.multiply(m2, wij_R)

     #     Здесь m0, m1, и m2 вычисляются как разность между значениями функции отображения камеры (CRFB, CRFG, CRFR для каждого цветового канала)
     #     и логарифмом времени экспозиции. Эти значения умножаются на соответствующие веса и добавляются к hdr, что помогает восстановить карту освещености.

     hdr = np.divide(hdr, wsum)
     hdr = np.exp(hdr)
     hdr = np.reshape(np.transpose(hdr), (self.row, self.col, 3))

     # Здесь hdr делится на сумму весов wsum, чтобы нормализовать результаты,
     # а затем применяется экспоненциальная функция для преобразования логарифмических значений обратно в освещенность.

     self.imgf32 = (hdr / np.amax(hdr) * 255).astype(np.float32)
     self.save_hdr_image()  # Сохраняем HDR-изображение перед тоновым отображением
     fig = plt.figure(constrained_layout=False, figsize=(10, 10))
     # plt.title("Карта освещенности HDR изображения", fontsize=20)
     plt.imshow(cv2.cvtColor(self.imgf32, cv2.COLOR_BGR2RGB))
     # plt.show()
     output_path = os.path.join('results', 'radianceMap.png')
     fig.savefig(output_path)

```
Метод recover_HDR_RadianceMap восстанавливает карту освещенности HDR, которая представляет собой внутреннее представление освещенности сцены.
Это позволяет сохранить всю информацию о свете в сцене, что является ключевым аспектом HDR-изображений.
```python
 def tone_mapping(self):

     hdr_image = cv2.imread('results/hdr_image.hdr', cv2.IMREAD_ANYDEPTH)

     # Применение тонирования для отображения на стандартном дисплее
     tonemap = cv2.createTonemap(3.0)  # Гамма-коррекция
     ldr_image = tonemap.process(hdr_image)

     # Отображение изображения
     plt.imshow(cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB))
     # plt.axis('off')  # Убрать оси координат
     # plt.show()
     # Сохранение отображенного изображения
     ldr_image = cv2.normalize(ldr_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
     output_path = os.path.join(app.config['RESULTS_FOLDER'], 'tone_mapping_ldr.jpg')
     cv2.imwrite(output_path, ldr_image)

     return ldr_image

```
Метод tone_mapping преобразует карту освещенности HDR в формат, пригодный для отображения на стандартных дисплеях.
Это включает в себя сжатие диапазона яркости и контраста, чтобы изображение выглядело естественно на не-HDR дисплеях.

```python
 def photograph_tone_mapping(self):

     hdr_image = cv2.imread('results/hdr_image.hdr', cv2.IMREAD_ANYDEPTH)

     # Преобразование HDR-изображения в градации серого
     lum = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)

     # Вычисление глобальной средней освещенности
     Lavg = np.exp(np.mean(np.log(lum + 1e-6)))

     # Применение фотографического метода тонового отображения
     a = 0.18  # Коэффициент масштабирования
     Ld = (a / Lavg) * lum
     Ld = Ld / (1 + Ld)  # Компрессия динамического диапазона

     # Применение отображенных тонов обратно к цветам
     ldr_image = np.zeros_like(hdr_image)
     for c in range(3):
         ldr_image[:, :, c] = hdr_image[:, :, c] * (Ld / lum)

     alpha = 0  # Коэффициент контраста
     beta = 400  # Яркость
     # Нормализация и преобразование в формат, подходящий для отображения
     ldr_image = cv2.normalize(ldr_image, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
     ldr_image = np.clip(ldr_image, 0, 255).astype('uint8')
     # Сохранение отображенного изображения
     output_path = os.path.join(app.config['RESULTS_FOLDER'], 'photo_ldr.jpg')
     cv2.imwrite(output_path, ldr_image)

     return ldr_image

```
Фотографический метод тонового отображения для преобразования HDR-изображения в LDR.
```python
 def photograph_tone_mapping(self):

     hdr_image = cv2.imread('results/hdr_image.hdr', cv2.IMREAD_ANYDEPTH)

     # Преобразование HDR-изображения в градации серого
     lum = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)

     # Вычисление глобальной средней освещенности
     Lavg = np.exp(np.mean(np.log(lum + 1e-6)))

     # Применение фотографического метода тонового отображения
     a = 0.18  # Коэффициент масштабирования
     Ld = (a / Lavg) * lum
     Ld = Ld / (1 + Ld)  # Компрессия динамического диапазона

     # Применение отображенных тонов обратно к цветам
     ldr_image = np.zeros_like(hdr_image)
     for c in range(3):
         ldr_image[:, :, c] = hdr_image[:, :, c] * (Ld / lum)

     alpha = 0  # Коэффициент контраста
     beta = 400  # Яркость
     # Нормализация и преобразование в формат, подходящий для отображения
     ldr_image = cv2.normalize(ldr_image, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
     ldr_image = np.clip(ldr_image, 0, 255).astype('uint8')
     # Сохранение отображенного изображения
     output_path = os.path.join(app.config['RESULTS_FOLDER'], 'photo_ldr.jpg')
     cv2.imwrite(output_path, ldr_image)

     return ldr_image

```

## Маршруты Flask-приложения

```python
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            exposure_times_strings = request.form.getlist('exposure_times')
            exposure_times = [float(time_str) for time_str in exposure_times_strings]
            session['exposure_times'] = exposure_times  # Сохраняем в сессии

            uploaded_files = request.files.getlist('images')
            filenames = []
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                filenames.append(file_path)
            session['filenames'] = filenames  # Сохраняем в сессии

        except Exception as e:
            return str(e), 400
        return redirect(url_for('process'))

    return render_template('index.html')

```
Основной маршрут. На странице выводится форма для получения изображений и веремен экспозиции, затем они сохраняются в сессии

```python
@app.route('/process', methods=['GET', 'POST'])
def process():
    try:
        # Извлекаем данные из сессии
        exposure_times_strings = request.form.get('exposure_times').split(',')
        exposure_times = []
        for time_str in exposure_times_strings:
            if '/' in time_str:
                numerator, denominator = time_str.split('/')
                exposure_times.append(float(numerator) / float(denominator))
            else:
                exposure_times.append(float(time_str))

        uploaded_files = request.files.getlist('images')
        filenames = []
        images = []
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            images.append(filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            filenames.append(file_path)

        # Обработка HDR
        mergeDebevec = HDR(app.config['UPLOAD_FOLDER'], filenames, exposure_times)
        postProcess = PostProcess(mergeDebevec)
        postProcess.process()
        postProcess.plot_ResponseCurves()
        postProcess.recover_HDR_RadianceMap()
        ldr_image1 = postProcess.tone_mapping()
        ldr_image2 = postProcess.photograph_tone_mapping()

        # Сохранение и отправка результата
        output_path = os.path.join(app.config['RESULTS_FOLDER'], 'tone_mapping_ldr.jpg')
        cv2.imwrite(output_path, ldr_image1)
        # Сохранение и отправка результата
        output_path = os.path.join(app.config['RESULTS_FOLDER'], 'photo_ldr.jpg')
        cv2.imwrite(output_path, ldr_image2)
        return render_template('process.html')
    except Exception as e:
        # Обработка исключений
        print(e)
        return str(e), 500

    # Если ни один из вышеуказанных путей не выполнен
    return render_template('process.html', {
                                            'filenames': images
                                            }
                           )

```
Маршрут на который попадает пользователь для преобразования изображений, тут происхходит основная логика проекта

```python
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory('uploads', filename)

```

Маршруты для получения директорий к файлам изображений