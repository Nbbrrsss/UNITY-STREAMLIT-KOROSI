import pandas as pd
import csv

def data_home() :
    data = pd.read_csv("dataset/data_lengkap_float_home.csv")
    return data

def df2() :
    data = pd.read_csv("dataset/data_training_testing.csv")
    return data

def data_unduh() :
    data = pd.read_excel("dataset/drugz_perez_raw.xlsx")
    return data

def dataframe_prediksi_data() :
    data = pd.read_excel("dataset/Perbandingan grafik data asli dengan algoritma.xlsx")
    return data