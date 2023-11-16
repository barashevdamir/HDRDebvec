# Здесь импортируются необходимые библиотеки и модули.
# Предупреждения отключаются для чистоты вывода, а также импортируется модуль time для отслеживания времени выполнения операций.
import cv2
import os
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import warnings
warnings.filterwarnings("ignore")
import time
from concurrent.futures import ThreadPoolExecutor


class HDR:
    def __init__(self, path, filenames, exposure_times):
        '''
        Определение класса HDR, который будет обрабатывать изображения для создания HDR-изображения.
        Конструктор загружает изображения, устанавливает времена экспозиции и выравнивает их с помощью алгоритма MTB (Median Threshold Bitmap).
        Класс HDR предназначен для создания изображений с высоким динамическим диапазоном (HDR) из серии фотографий с разной экспозицией.
        Это достигается путем объединения информации о свете и тени из разных изображений для создания одного изображения с более широким диапазоном яркости.

        '''
        with ThreadPoolExecutor() as executor:
            self.images = list(executor.map(lambda fn: cv2.imread(''.join([path, fn])), filenames))

        self.times = np.array(exposure_times, dtype=np.float32)
        self.N = len(self.images)
        self.row = len(self.images[0])
        self.col = len(self.images[0][0])

        # Выравнивание исходных изображений
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(self.images, self.images)

        # Коэффициент веса для плавности кривых отклика
        self.l = 10

    def display_OriginalImages(self, figureSize):
        '''
        Метод display_OriginalImages используется для визуализации исходных изображений с разной экспозицией.
        Это помогает визуально оценить, какие области каждого изображения хорошо экспонированы, а какие нет.

        '''
        offset = 50
        Canvas = np.ones((self.row, (self.col + offset) * self.N, 3), dtype=np.float32)

        for l in range(self.N):
            c1 = l * (self.col + offset)
            c2 = c1 + self.col
            Canvas[0:self.row, c1:c2, :] = (self.images[l] / 255).astype(np.float32)

        fig = plt.figure(constrained_layout=False, figsize=figureSize)
        plt.imshow(cv2.cvtColor(Canvas[:, :c2, :], cv2.COLOR_BGR2RGB))
        plt.title("Исходные изображения", fontsize=20)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        fig.savefig('originals.png')

    def weightingFunction(self):
        '''
        Метод weightingFunction создает весовую функцию, которая придает больший вес хорошо экспонированным пикселям и меньший вес плохо экспонированным.
        Это важно для минимизации шума и артефактов в конечном HDR-изображении.

        '''
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

    def samplingPixelValues(self):
        '''
        Метод samplingPixelValues выбирает пиксели из каждого изображения для определения функции отклика камеры.
        Это ключевой шаг в процессе создания HDR, так как функция отклика камеры необходима для правильного объединения различных экспозиций.

        '''
        # Определение количества образцов для выборки на основе количества изображений и диапазона значений пикселей.
        numSamples = math.ceil(255 * 2 / (self.N - 1)) * 2  # Количество образцов для выборки.
        numPixels = self.row * self.col  # Общее количество пикселей в изображении.

        # Определение шага выборки для равномерного распределения выборки по всему изображению.
        step = int(np.floor(numPixels / numSamples))  # Шаг выборки пикселей.
        self.sampleIndices = list(range(0, numPixels, step))[:-1]  # Индексы выбранных пикселей.

        # Создание плоского (одномерного) представления изображений для упрощения доступа к пикселям.
        self.flattenImage = np.zeros((self.N, 3, numPixels), dtype=np.uint8)  # Массив для плоских изображений.

        def flatten_channel(channel_index):
            for i in range(self.N):
                # Преобразование каждого цветового канала в одномерный массив.
                self.flattenImage[i, channel_index] = np.reshape(self.images[i][:, :, channel_index], (numPixels,))

        # Использование многопоточности для ускорения процесса плоского представления изображений
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(flatten_channel, range(3))

        # Логарифмирование времен экспозиции для использования в математических расчетах.
        X = np.log(self.times)  # Логарифмические времена экспозиции.

        # Массивы для хранения значений пикселей в выбранных местах
        self.ZG = np.zeros((numSamples, self.N), dtype=np.uint8)
        self.ZB = np.zeros((numSamples, self.N,), dtype=np.uint8)
        self.ZR = np.zeros((numSamples, self.N), dtype=np.uint8)

        # Получение выбранных значений пикселей
        for k in range(self.N):
            self.ZB[:, k] = self.flattenImage[k, 0][self.sampleIndices]
            self.ZG[:, k] = self.flattenImage[k, 1][self.sampleIndices]
            self.ZR[:, k] = self.flattenImage[k, 2][self.sampleIndices]

        # Отсев выборок, где значения пикселей уменьшаются с увеличением экспозиции (что не логично).
        '''
        Отсев выборок, где значения пикселей уменьшаются с увеличением экспозиции, считается нелогичным,
        потому что в нормальных условиях, когда экспозиция увеличивается (то есть, когда на сенсор камеры попадает больше света),
        ожидается, что значения пикселей также увеличатся. Это основное предположение фотографии: если вы увеличиваете время,
        в течение которого свет воздействует на пиксель (экспозиция), то пиксель должен стать ярче, что означает увеличение его значения.
        '''
        ind = np.arange(0, numSamples)  # Индексы всех выборок.
        idx = []  # Список для хранения индексов выборок, которые нужно исключить.
        for i in range(numSamples):
            for k in range(self.N - 1):
                if self.ZG[i, k] > self.ZG[i, k + 1]:
                    idx.append(i)  # Добавление индекса для исключения.
                    break
        ind = np.delete(ind, idx, 0)  # Исключение неподходящих выборок.

        # Обновление массивов значений пикселей после исключения неподходящих выборок.
        self.ZB = self.ZB[ind]
        self.ZG = self.ZG[ind]
        self.ZR = self.ZR[ind]

        # Массивы для хранения логарифмических времен экспозиции
        r, c = self.ZG.shape[:2]
        self.Bij = np.tile(np.log(self.times), (self.row * self.col, 1))

    def CRFsolve(self, Z):
        '''
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
        '''

        n = 256

        s1, s2 = Z.shape
        A = np.zeros((s1 * s2 + n + 1, n + s1))
        b = np.zeros((A.shape[0], 1))

        # Включение уравнений для соответствия данным
        k = 0
        for i in range(s1):
            for j in range(s2):
                wij = self.w[Z[i, j]]
                A[k, Z[i, j]] = wij
                A[k, n + i] = -wij
                b[k] = wij * self.Bij[i, j]
                k += 1

        # Фиксация кривой путем установки ее среднего значения в ноль
        A[k, 129] = 0
        k += 1

        # Включение уравнений для плавности
        for i in range(1, n - 2):
            A[k, i] = self.l * self.w[i + 1]
            A[k, i + 1] = -2 * self.l * self.w[i + 1]
            A[k, i + 2] = self.l * self.w[i + 1]
            k += 1
        # Решение системы с использованием метода сингулярного разложения (SVD)
        x = np.linalg.lstsq(A, b)
        x = x[0]
        CRF = x[0:n]
        lE = x[n: len(x)]

        return CRF, lE

    def plot_ResponseCurves(self):
        '''
        Метод plot_ResponseCurves отображает графики функций отклика для каждого цветового канала.
        Эти кривые показывают, как каждый пиксель камеры реагирует на различные уровни освещенности.

        '''
        px = list(range(0, 256))
        fig = plt.figure(constrained_layout=False, figsize=(5, 5))
        plt.title("Кривые отклика", fontsize=20)
        plt.plot(px, np.exp(self.gR), 'r')
        plt.plot(px, np.exp(self.gB), 'b')
        plt.plot(px, np.exp(self.gG), 'g')
        plt.ylabel("log(X)", fontsize=20)
        plt.xlabel("Значение пикселя", fontsize=20)
        plt.show()
        fig.savefig('curvesCRF.png')

    def process(self):
        '''
        Вызывает предыдущие методы для построения функции отклика и подготовки данных для восстановления карты яркости HDR.
        '''
        self.weightingFunction()
        self.samplingPixelValues()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.CRFsolve, self.ZR),
                       executor.submit(self.CRFsolve, self.ZG),
                       executor.submit(self.CRFsolve, self.ZB)]

            self.gR, self.lER = futures[0].result()
            self.gG, self.lEG = futures[1].result()
            self.gB, self.lEB = futures[2].result()


class PostProcess(HDR):
    '''
    Класс PostProcess наследует HDR и добавляет методы для сохранения и тонового отображения HDR-изображения.
    Тоновое отображение необходимо, потому что стандартные дисплеи не могут отображать HDR напрямую.

    '''
    def __init__(self, HDR):
        super().__init__(path, filenames, exposure_times)

    def save_hdr_image(self, filename='hdr_image.hdr'):
        '''
        Метод save_hdr_image сохраняет HDR-изображение в формате, который может быть использован другими программами и устройствами, поддерживающими HDR.

        '''
        cv2.imwrite(filename, self.imgf32)

    def recover_HDR_RadianceMap(self):
        '''
        Метод recover_HDR_RadianceMap восстанавливает карту освещенности HDR, которая представляет собой внутреннее представление освещенности сцены.
        Это позволяет сохранить всю информацию о свете в сцене, что является ключевым аспектом HDR-изображений.

        '''
        m = np.zeros((self.flattenImage.shape[1:]))
        wsum = np.zeros(self.flattenImage.shape[1:])
        hdr = np.zeros(self.flattenImage.shape[1:])

        lnDt = np.log(self.times)

        for i in range(self.N):
            wij_B = self.w[self.flattenImage[i, 0]]
            wij_G = self.w[self.flattenImage[i, 1]]
            wij_R = self.w[self.flattenImage[i, 2]]

            wsum[0, :] += wij_B
            wsum[1, :] += wij_G
            wsum[2, :] += wij_R

            m0 = np.subtract(self.gB[self.flattenImage[i, 0]], lnDt[i])[:, 0]
            m1 = np.subtract(self.gG[self.flattenImage[i, 1]], lnDt[i])[:, 0]
            m2 = np.subtract(self.gR[self.flattenImage[i, 2]], lnDt[i])[:, 0]

            hdr[0] = hdr[0] + np.multiply(m0, wij_B)
            hdr[1] = hdr[1] + np.multiply(m1, wij_G)
            hdr[2] = hdr[2] + np.multiply(m2, wij_R)

        hdr = np.divide(hdr, wsum)
        hdr = np.exp(hdr)
        hdr = np.reshape(np.transpose(hdr), (self.row, self.col, 3))

        self.imgf32 = (hdr / np.amax(hdr) * 255).astype(np.float32)
        self.save_hdr_image()  # Сохраняем HDR-изображение перед тоновым отображением
        fig = plt.figure(constrained_layout=False, figsize=(10, 10))
        plt.title("Карта яркости HDR изображения", fontsize=20)
        plt.imshow(cv2.cvtColor(self.imgf32, cv2.COLOR_BGR2RGB))
        plt.show()
        fig.savefig('radianceMap.png')



    def tone_mapping(self):
        '''
        Метод tone_mapping преобразует карту радианса HDR в формат, пригодный для отображения на стандартных дисплеях.
        Это включает в себя сжатие диапазона яркости и контраста, чтобы изображение выглядело естественно на не-HDR дисплеях.
        '''
        hdr_image = cv2.imread('hdr_image.hdr', cv2.IMREAD_ANYDEPTH)

        # Применение тонирования для отображения на стандартном дисплее
        tonemap = cv2.createTonemap(3.0)  # Гамма-коррекция
        ldr_image = tonemap.process(hdr_image)

        # Отображение изображения
        plt.imshow(cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Убрать оси координат
        plt.show()
        # Сохранение отображенного изображения
        ldr_image = cv2.normalize(ldr_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite('tone_mapping_ldr.jpg', ldr_image)

        return ldr_image

    def photograph_tone_mapping(self):
        '''
        Фотографический метод тонового отображения для преобразования HDR-изображения в LDR.
        '''
        hdr_image = cv2.imread('hdr_image.hdr', cv2.IMREAD_ANYDEPTH)

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
        # Отображение результата
        plt.imshow(cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Убрать оси координат
        plt.show()
        # Сохранение отображенного изображения
        cv2.imwrite('photo_ldr.jpg', ldr_image)

        return ldr_image


# Создается экземпляр класса HDR,
# отображаются оригинальные изображения,
# обрабатываются для создания HDR-изображения,
# отображается функция отклика камеры,
# восстанавливает карту яркости и применяется тоновое отображение.
# Время выполнения каждого этапа измеряется и выводится.
#
# path = "uploads/"
# filenames = ['image0.jpg', 'image1.jpg', 'image2.jpg']
# exposure_times = [1/6.0, 1.3, 5.0]
# mergeDebevec = HDR(path, filenames, exposure_times)
# mergeDebevec.display_OriginalImages(figureSize=(20,20))
# start_time = time.time()
# postProcess = PostProcess(mergeDebevec)
# postProcess.process()
# postProcess.plot_ResponseCurves()
# print(time.time() - start_time)
# start_time = time.time()
# postProcess.recover_HDR_RadianceMap()
# print(time.time() - start_time)
# ldr_image = postProcess.tone_mapping()
# ldr_image = postProcess.photograph_tone_mapping()
