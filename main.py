# ============== Import Library ===================
# ============== Import Library ===================
import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64
from sklearn.model_selection import train_test_split
import re  # Import library regular expression untuk manipulasi teks
import random  # Import library random untuk pemilihan respons acak
import json  # Import library json untuk membaca data dari file JSON

import dataset.path_data as dth

from function_module.machine import base_machine_learning, analisis_model_terbaik
from function_module.analisis_data import analisis
from function_module.chatbot import chatbot
# ============== End of Import Library ============

st.set_page_config(
    page_title="Website Inhibitor Korosi",
    page_icon="labmatics icon.ico"
)

# ============== Load Data Set =====================

df = dth.data_home()
df2 = dth.df2()
data_unduh = dth.data_unduh()

# # Memuat data dari JSON

# ============= End Of Load DataSet ================


with st.sidebar:
    st.image('labmatics.png', width=350)
    menuweb = st.radio(
        "Menu Website", ["Home", "ChatBot", "Machine Learning Ops","Analisis Data"])

st.image("Group 10.png", width=350)
if menuweb == "Home":
    st.title("Senyawa Obat Inhibitor Korosi")
    st.markdown("Senyawa inhibitor korosi adalah zat atau senyawa kimia yang ditambahkan ke lingkungan yang berpotensi mengakibatkan korosi untuk mencegah atau mengurangi laju korosi logam atau material lain. Inhibitor korosi bertujuan untuk melindungi permukaan logam dari oksidasi atau reaksi kimia lain yang dapat mengurangi kekuatannya atau mempersingkat umur pemakaiannya.")

    tab1, tab2, tab3 = st.tabs(
        ['Kamus Senyawa', 'Filter Senyawa', "Unduh Dataset"])

    with tab1:

        st.header("Kamus Senyawa")
        kolom_filter = {
            'Common_name': 'Nama Senyawa',
            'Formula': 'Rumus Senyawa',
            'IUPAC_name': 'Tata nama senyawa kimia',
        }

        to_filter_columns = st.selectbox("Masukkan filter Senyawa untuk ditampilkan", options=list(
            kolom_filter.keys()), format_func=lambda x: kolom_filter[x])

        text_search = st.text_input("Cari informasi tentang senyawa", value="")
        text_search = text_search.lower()

        df["Common_name"] = df["Common_name"].str.lower()
        df["Formula"] = df["Formula"].str.lower()

        N_cards_per_row = 1
        if text_search:
            # Filter the dataframe using masks
            nama_column = "Common_name"
            if to_filter_columns:
                nama_column = to_filter_columns

            m1 = df[nama_column].str.contains(text_search)
            df_search = df[m1]

            for n_row, row in df_search.reset_index().iterrows():
                i = n_row % N_cards_per_row
                if i == 0:
                    st.write("---")
                    cols = st.columns(N_cards_per_row, gap="large")
                # draw the card
                with cols[n_row % N_cards_per_row]:
                    st.markdown(
                        f"[**{row['Common_name'].strip().capitalize()}**]({row['Ref']})")
                    st.markdown(
                        f"{row['Common_name'].capitalize()} adalah sebuah senyawa dengan Rumus Kimia **{row['Formula'].upper()}**.Senyawa dengan Tata Nama Kimia **{row['IUPAC_name']}** ini memiliki Persentase Energi Ionisasi sebesar **{round(row['IE EXP (%)'],2)}%**")
                    st.markdown(f"Berat molekul dari senyawa ini adalah {row['Molecular_weight MW (g/mol)']} g/mol. {row['Common_name'].capitalize()} memiliki kemampuan untuk membentuk ikatan hidrogen, dengan Hydrogen Acceptor Count sebanyak {row['Hydrogen Acceptor Count']} dan Hydrogen Donor Count sebanyak {row['Hydrogen Donor Count']}. Selain itu, kemampuan atom dalam molekul ini untuk menarik pasangan elektron sebesar {row['Electronegativity (eV)']} dan ukuran kestabilan molekulnya juga mencapai {row['Electronegativity (eV)']}.")
            else:
                st.write("Senyawa tidak ditemukan!")
        else:
            N_cards_per_row = 3
            for n_row, row in df.reset_index().iterrows():
                i = n_row % N_cards_per_row
                if i == 0:
                    st.write("---")
                    cols = st.columns(N_cards_per_row, gap="large")
                    # draw the card
                with cols[n_row % N_cards_per_row]:
                    st.markdown(
                        f"[**{row['Common_name'].strip().capitalize()}**]({row['Ref']})")
                    st.markdown(
                        f"**{row['Formula'].upper()}**")
                    st.markdown(f"**{row['IUPAC_name']}**")
                    st.markdown(
                        f"Persentase Energi Ionisasi **{round(row['IE EXP (%)'],2)}%**")
    with tab2:
        st.error('Perhatian! Data IE EXP (Persentase eksperimental dari efisiensi penghambatan korosi) yang kami sajikan merupakan hasil dari gabungan prediksi dari model Machine Learning yang telah kami buat.Jika ingin melihat dataset asli bisa download pada Tab "Unduh Dataset" ')

        st.header("Filter Senyawa")

        kolom_filter = {
            'Molecular_weight MW (g/mol)': 'Berat Molekul (g/mol)',
            'pKa': 'Kekuatan Asam',
            'Log P': 'Kelarutan Lemak',
            'Log S': 'Kelarutan Dalam Air',
            'Polar Surface Area': 'Luas permukaan molekul',
            'Polarizability': 'Polarisabilitas',
            'HOMO (eV)': 'Energi orbital molekul yang terisi',
            'LUMO (eV)': 'Energi orbital molekul yang tidak terisi',
            'Electronegativity (eV)': 'Electronegativity (eV)',
            ' ?N_Fe ': 'Indeks daya tunduk molekul',
            'IE EXP (%)': 'Efisiensi Anti Korosi',
        }

        # add filter senyawa by kategori
        kolom_filtertab2 = {
            'all': 'Semua',
            'Drugs': 'Senyawa Drugs'
        }

        # default kolom all
        filter_kategori2 = "all"

        col_filter3, col_filter4 = st.columns(2)

        def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            to_filter_columns = st.multiselect("Masukkan filter Senyawa untuk ditampilkan", options=list(
                kolom_filter.keys()), format_func=lambda x: kolom_filter[x])

            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")

                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Ukuran {column} senyawa",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df[column] = df[column].astype(float)
                df = df[df[column].between(*user_num_input)]
                df.insert(1, column, df.pop(column))
                # Mengurutkan DataFrame berdasarkan kolom yang diinginkan
                df = df.sort_values(by=column, ascending=False)

            return df

        df["Common_name"] = df["Common_name"].str.capitalize()
        if to_filter_columns:
            st.dataframe(filter_dataframe(df))
        else:
            st.dataframe(df)

    with tab3:
        st.download_button(
            label="Download Dataset",
            data=data_unduh.to_csv().encode('utf-8'),
            file_name='Data.csv',
            mime='text/csv',
        )

        st.dataframe(data_unduh)

# ==================================== Chatbot ====================================================================================
if menuweb == "ChatBot":
    chatbot()

# ========================= Halaman App ========================================================================================================================================================================================================================================================================================================================================================================
if menuweb == "Machine Learning":
    # Streamlit App Header
    st.title("Prediksi IE% Korosi dalam senyawa")
    st.markdown(
        "Silakan prediksi Efisiensi Korosi Senyawa menggunakan parameter-parameter yang ada!")
    st.error('Berdasarkan eksperimen, Model Terbaik lebih disarankan untuk memprediksi Senyawa dengan baik karena memiliki nilai akurasi yang tinggi dan nilai error yang rendah. Kombinasi model yang lain sebaiknya hanya digunakan sebagai referensi atau mungkin hanya menjadi pembanding.')
    opsi_model = st.selectbox(
        "", ['Experiment Machine Learning','Analisis Model Terbaik'])

    st.write("---")

    if opsi_model == "Experiment Machine Learning":
        base_machine_learning()
    else:
        analisis_model_terbaik()


# ========================= Ahkir halaman App ===================================================================================================================================================================================================================================================================================================================================================================

# Data Analisis
if menuweb == "Analisis Data":
    st.title("Data Set")
    analisis()
