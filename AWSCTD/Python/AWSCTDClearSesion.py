# Файл: AWSCTDClearSesion.py (оновлений)
import tensorflow as tf
import gc

# Скидає стан Keras/TF, очищує пам'ять графу та вивільняє GPU-пам'ять (якщо можливо)
def reset_keras():
    tf.keras.backend.clear_session()
    gc.collect()
    # Налаштування параметрів GPU, якщо потрібен контроль (необов'язково):
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        pass

