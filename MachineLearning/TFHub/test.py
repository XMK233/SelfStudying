import sys
###
from multiprocessing import Pool, Manager
import os, time, random, psutil, pynvml, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## we don't have enough GPU mem. Very sad.
###
import itertools

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

MODEL_SAVED_DIR = r"J:\Finetuned_ImageNet_Model" # os.path.join(os.getcwd(), "models")     #
SNAPSHOT_INTERVAL = 5 ## in seconds
###
from random import choice, sample
import shutil
###
def get_global_usage_percentage(model_name):
    model_dir = os.path.join(MODEL_SAVED_DIR, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_file = os.path.join(model_dir, "log.txt")
    with open(log_file, "a") as lf:
        ## GPU memory:
        lf.write("----------This is divider---------------\n")
        _ = os.popen('"C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"').read().strip() # os.popen('nvidia-smi').read().strip() #
        lf.write(_)
        lf.write("\n")
        ## CPU usage
        lf.write("cpu percentage: {}\n".format(psutil.cpu_percent(1)))
        ## memory usage
        lf.write("memory: {}\n".format(psutil.virtual_memory()))
        ## disk usage
        lf.write("disk io counter: {}\n".format(psutil.disk_io_counters()))
        lf.write("disk usage for disk C: {}\n".format(psutil.disk_usage(r"C:")))
        lf.write("disk usage for cwd disk: {}\n".format(psutil.disk_usage(os.getcwd())))
    return

def keep_getting_stats(queue, model_name):
    while True:
        if queue.qsize() != 0:
            break
        get_global_usage_percentage(model_name)
        time.sleep(SNAPSHOT_INTERVAL)

def generate_datasets(IMAGE_SIZE, BATCH_SIZE):
    data_dir = r"C:\Users\xmk233\.keras\datasets\flower_photos" # "/root/.keras/datasets/flower_photos" #
    data_dir_validation = data_dir + "-validation"
    data_dir_test = data_dir + "-test"

    datagen_kwargs = dict(rescale=1./255) # , validation_split=0
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                    interpolation="bilinear")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs).flow_from_directory(
        data_dir, shuffle=False, **dataflow_kwargs)

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs).flow_from_directory(
        data_dir_validation, shuffle=False, **dataflow_kwargs)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs).flow_from_directory(
        data_dir_test, shuffle=False, **dataflow_kwargs)

    return train_datagen, validation_datagen, test_datagen