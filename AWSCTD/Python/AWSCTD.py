import time
import sys
# Вставляємо шлях до утиліт (папка Utils)
sys.path.insert(1, 'Utils')

if len(sys.argv) != 3:
    print("Parameters example: AWSCTD.py file_to_data.csv CNN")
    quit()

import os
# Підключаємо TensorFlow (включно з Keras)
import tensorflow as tf
# Прибрати надлишкові попередження TensorFlow (опційно):
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Отримуємо аргументи командного рядка
m_sDataFile = sys.argv[1]
m_sModel   = sys.argv[2]

import numpy as np
np.random.seed(0)  # Фіксуємо random seed для відтворюваності

# Імпортуємо модулі проєкту
import AWSCTDReadData
import AWSCTDCreateModel
import AWSCTDClearSesion
import gc

# Отримуємо робочу директорію і читаємо конфігурацію
m_sWorkingDir = os.getcwd() + '/'
print(("Working directory:", m_sWorkingDir))
import configparser
config = configparser.ConfigParser()
config.read(m_sWorkingDir + 'config.ini')
nEpochs      = config.getint('MAIN', 'nEpochs')      # Кількість епох (напр. 200)
nBatchSize   = config.getint('MAIN', 'nBatchSize')   # Розмір батчу (напр. 100)
nPatience    = config.getint('MAIN', 'nPatience')    # Раннє завершення, якщо немає поліпшень
nKFolds      = config.getint('MAIN', 'nKFolds')      # Кількість фолдів для крос-валідації (напр. 5)
bCategorical = config.getboolean('MAIN', 'bCategorical') # true, якщо багатокласова класифікація

# Виводимо зчитаний конфіг для перевірки
with open(m_sWorkingDir + 'config.ini', 'r') as fIniFile:
    sConfig = fIniFile.read()
print("Config file:")
print(sConfig)

# Завантажуємо дані
Xtr, Ytr, m_nParametersCount, m_nClassCount, m_nWordCount = \
    AWSCTDReadData.ReadDataImpl(m_sDataFile, bCategorical)
print(("Loaded Y shape:", Ytr.shape))  # перевірка завантажених міток

gc.collect()  # прибирання сміття перед навчанням моделей

# Визначаємо функцію для поступового зменшення learning rate (необов'язково використовувати)
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.8
    epochs_drop = 100.0
    # альтернативна формула спадання learning rate:
    lrate = initial_lrate * 1.0 / (1.0 + (0.0000005 * epoch * 100))
    return 0.0001 if lrate < 0.0001 else lrate

# Налаштовуємо колбеки: рання зупинка (EarlyStopping) і (опційно) адаптивний learning rate
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
# Моніторимо точність (accuracy) на тренуванні; patience визначено в конфігу
sMonitor = 'accuracy'
# Якщо багатокласова класифікація, Keras теж обчислює 'accuracy' (вона інтерпретується як categorical_accuracy).
# За потреби можна було б явно використовувати 'categorical_accuracy', але це не потрібно.
early_stop = EarlyStopping(monitor=sMonitor, patience=nPatience, verbose=1)
# scheduler не використано, але можна додати: lrate = LearningRateScheduler(step_decay, verbose=1)

# Підготовка до крос-валідації
from sklearn.model_selection import KFold
kfold = KFold(n_splits=nKFolds, shuffle=True)
nFoldNumber = 1

# Масиви для збору результатів по фолдам
arrAcc, arrLoss = [], []
arrTimeFit, arrTimeTest, arrTimePredict = [], [], []

# Мітрики та засоби оцінки
from sklearn.metrics import confusion_matrix, roc_curve, auc
from numpy import interp  # використовуємо numpy для інтерполяції ROC-кривої
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'  # щоб текст на .svg залишався текстом, а не кривими

# Підготовка структур для ROC-кривих (для 2 або більше класів)
tprs, aucs, EER = {}, {}, {}
if m_nClassCount == 2:
    tprs = {0: [], 1: []}
    aucs = {0: [], 1: []}
    EER  = {0: [], 1: []}
elif m_nClassCount == 5:
    # ... аналогічно для 5 класів ...
    tprs = {0: [], 1: [], 2: [], 3: [], 4: []}
    # ... (пропущено задля стислості) ...
elif m_nClassCount == 6:
    # ... для 6 класів (5 типів шкідників + 1 клас "Benign") ...
    tprs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    # ... (пропущено) ...
mean_fpr = np.linspace(0, 1, 100)

# Цикл по фолдам крос-валідації
for train, test in kfold.split(Xtr, Ytr):
    print(f"KFold number: {nFoldNumber}")
    nFoldNumber += 1

    # Створюємо нову модель для поточного фолду
    model = AWSCTDCreateModel.CreateModelImpl(
        m_sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical
    )

    # Навчання моделі на тренувальній частині фолду
    startFit = time.time()
    history = model.fit(Xtr[train], Ytr[train],
                        epochs=nEpochs, batch_size=nBatchSize,
                        callbacks=[early_stop], verbose=1)
    endFit = time.time()
    arrTimeFit.append(endFit - startFit)
    # Зберігаємо історію навчання (для графіків точності/втрат)
    # model_history.append(history)  # якщо потрібно накопичувати історію

    # Оцінка моделі на тестовій частині фолду
    startTest = time.time()
    scores = model.evaluate(Xtr[test], Ytr[test], verbose=0)
    endTest = time.time()
    arrTimeTest.append(endTest - startTest)

    # Прогноз для тестової вибірки (для побудови ROC та обчислення дод.метрик)
    startPredict = time.time()
    y_pred = model.predict(Xtr[test])
    endPredict = time.time()
    arrTimePredict.append(endPredict - startPredict)
    # Накопичуємо кількість тестових зразків (для середнього часу прогнозу)
    # ...

    # Виводимо основну метрику точності для даного фолду
    print((model.metrics_names))  # назви метрик (['loss', 'accuracy'] очікувано)
    print(f"{model.metrics_names[1]}: {scores[1]*100:.2f}%")

    arrAcc.append(scores[1] * 100)
    arrLoss.append(scores[0])

    # "Sanity check" точності: обчислення вручну відсотку правильних класифікацій
    nPredCorr = 0
    if bCategorical:
        # Якщо one-hot мітки – порівнюємо argmax
        all_count = len(y_pred)
        for i in range(all_count):
            if np.argmax(Ytr[test][i]) == np.argmax(y_pred[i]):
                nPredCorr += 1
        dAccuracy = float(nPredCorr) / all_count * 100.0
    else:
        # Якщо двокласовий випадок – бінаризуємо й порівнюємо
        y_pred_class = (y_pred > 0.5).astype(int)
        all_count = len(y_pred_class)
        for i in range(all_count):
            if Ytr[test][i] == y_pred_class[i]:
                nPredCorr += 1
        dAccuracy = float(nPredCorr) / all_count * 100.0
    print(f"Accuracy (Sanity Check): {dAccuracy:.2f}%")

    # Оновлення матриці змішування (confusion matrix) та ROC-метрик
    if m_nClassCount != 2 or bCategorical:
        # Для багатокласового випадку: беремо argmax
        y_pred_labels = np.argmax(y_pred, axis=1)
        true_labels = Ytr[test].argmax(axis=1) if bCategorical else Ytr[test]
    else:
        # Для двокласового випадку з мітками 0/1
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        true_labels = Ytr[test].flatten()
    cm = confusion_matrix(true_labels, y_pred_labels)
    # (Накопичення cm між фолдами можна виконати, якщо потрібно усереднити результати)

    # ROC-криві та показник EER для кожного класу (тільки якщо bCategorical і більше 1 класу)
    if bCategorical:
        for x in range(m_nClassCount):
            fpr, tpr, thresholds = roc_curve(Ytr[test][:, x], y_pred[:, x])
            fnr = 1 - tpr
            # Обчислюємо точку EER (Equal Error Rate)
            eer_ = fpr[np.nanargmin(np.absolute(fnr - fpr))]
            EER[x].append(eer_)
            # Інтерполюємо ROC-криву до mean_fpr для усереднення
            tprs[x].append(interp(mean_fpr, fpr, tpr))
            tprs[x][-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs[x].append(roc_auc)
            # (Побудова графіку ROC для кожного фолду опущена задля стислості)
    # Прибирання моделі і очищення сесії після кожного фолду
    try:
        del model
    except:
        pass
    AWSCTDClearSesion.reset_keras()

# Після завершення всіх фолдів – зведена статистика часу і точності
total_time = sum(arrTimeFit) + sum(arrTimeTest) + sum(arrTimePredict)
print(f"All time : {total_time:.7f}")
print(f"Training time (avg per fold): {np.mean(arrTimeFit):.7f}")
print(f"Testing time (avg per fold): {np.mean(arrTimeTest):.7f}")
print(f"Predicting time (total): {sum(arrTimePredict):.7f}")
print(f"Predicting time (per sample): {(sum(arrTimePredict)/len(Ytr)):.7f}")

print((" Acc: %.2f%% (+/- %.2f%%)" % (np.mean(arrAcc), np.std(arrAcc))))

# Збереження результатів у базу SQLite
import sqlite3
con = sqlite3.connect('results.db')
sTestTag = m_sModel
# Формуємо запис з усіма потрібними полями (як в оригінальному коді)
result = (m_sDataFile, m_nParametersCount, m_nClassCount, nEpochs, nBatchSize,
          str(model.to_json()), total_time, np.mean(arrAcc), np.mean(arrLoss),
          np.mean(arrTimeFit), np.mean(arrTimeTest), sTestTag, np.std(arrAcc),
          np.std(arrLoss), sTime, (sum(arrTimePredict)/len(Ytr)),
          # точності на кожному фолді:
          (arrAcc[0] if len(arrAcc)>0 else None),
          (arrAcc[1] if len(arrAcc)>1 else None),
          (arrAcc[2] if len(arrAcc)>2 else None),
          (arrAcc[3] if len(arrAcc)>3 else None),
          (arrAcc[4] if len(arrAcc)>4 else None),
          sConfig)
sql = """INSERT INTO results
    (File, ParamCount, ClassCount, Epochs, BatchSize, Model, Time, Acc, Loss,
     TimeTrain, TimeTest, Comment, AccStd, LossStd, ExecutionTime, PredictingOneTime,
     Acc1, Acc2, Acc3, Acc4, Acc5, Config)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
cur = con.cursor()
cur.execute(sql, result)
con.commit()
con.close()

# Побудова і збереження графіків навчання і матриці конфузії
import AWSCTDPlotAcc, AWSCTDPlotCM
AWSCTDPlotAcc.plot_acc_loss(history, m_sModel, m_sDataFile, bCategorical, m_sWorkingDir)
AWSCTDPlotCM.plot_cm(cm, m_sModel, m_nClassCount, m_sDataFile, m_sWorkingDir)
# (Побудову ROC-графіків теж можна додати за потреби)

