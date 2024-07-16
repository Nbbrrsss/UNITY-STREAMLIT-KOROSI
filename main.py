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
    page_title="Corrosion Inhibitor Website",
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
        "Website Menu", ["Home", "ChatBot", "Anti-corrosion Efficiency","Data Analysis"])

st.image("Group 10.png", width=350)
if menuweb == "Home":
    st.title("Corrosion Inhibitor Drugs Compounds")
    st.markdown("Corrosion inhibitor compounds are substances or chemical compounds that are added to a potentially corrosive environment to prevent or reduce the rate of corrosion of metals or other materials. Corrosion inhibitors aim to protect metal surfaces from oxidation or other chemical reactions that can reduce their strength or shorten their service life.")

    tab1, tab2, tab3 = st.tabs(
        ['Compound Dictionary', 'Compound Filter', "Download Dataset"])

    with tab1:

        st.header("Compound Dictionary")
        kolom_filter = {
            'Common_name': 'Compound Name',
            'Formula': 'Compound Formula',
            'IUPAC_name': 'Chemical compound name system',
        }

        to_filter_columns = st.selectbox("Input Compound Filter to display", options=list(
            kolom_filter.keys()), format_func=lambda x: kolom_filter[x])

        text_search = st.text_input("Search for information about compounds", value="")
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
                        f"{row['Common_name'].capitalize()} is a compound with Chemical Formula **{row['Formula'].upper()}**.Compounds with Chemical Names **{row['IUPAC_name']}** it has an Ionization Energy Percentage of **{round(row['IE EXP (%)'],2)}%**")
                    st.markdown(f"The molecular weight of this compound is {row['Molecular_weight MW (g/mol)']} g/mol. {row['Common_name'].capitalize()} has the ability to form hydrogen bonds, with Hydrogen Acceptor Count as much as {row['Hydrogen Acceptor Count']} and Hydrogen Donor Count as much as {row['Hydrogen Donor Count']}. In addition, the ability of the atoms in this molecule to attract electron pairs as large as {row['Electronegativity (eV)']} and the size of the molecular stability also reaches {row['Electronegativity (eV)']}.")
            else:
                st.write("Compound not found!")
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
                        f"Ionization Energy Percentage **{round(row['IE EXP (%)'],2)}%**")
    with tab2:
        st.error('Attention! The IE EXP (Experimental percentage of corrosion inhibition efficiency) data we present is the result of combined predictions from the Machine Learning models we have created. If you want to see the original dataset, you can download it from the “Download Dataset” tab." ')

        st.header("Compound Filter")

        kolom_filter = {
            'Molecular_weight MW (g/mol)': 'Molecular Weight (g/mol)',
            'pKa': 'Acid Strength',
            'Log P': 'Fat Solubility',
            'Log S': 'Solubility in Water',
            'Polar Surface Area': 'Molecular surface area',
            'Polarizability': 'Polarizability',
            'HOMO (eV)': 'Energy of filled molecular orbitals',
            'LUMO (eV)': 'Unfilled molecular orbital energy',
            'Electronegativity (eV)': 'Electronegativity (eV)',
            ' ?N_Fe ': 'Index of the molecule\'s resistivity',
            'IE EXP (%)': 'Anti-corrosion efficiency',
        }

        # add Compound Filter by kategori
        kolom_filtertab2 = {
            'all': 'All',
            'Drugs': 'Compound Drugs'
        }

        # default kolom all
        filter_kategori2 = "all"

        col_filter3, col_filter4 = st.columns(2)

        def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            to_filter_columns = st.multiselect("Insert Compound Filter to display", options=list(
                kolom_filter.keys()), format_func=lambda x: kolom_filter[x])

            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")

                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Size of {column} compound",
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
if menuweb == "Anti-corrosion Efficiency":
    # Streamlit App Header
    st.title("Predicted IE% Corrosion in compounds")
    st.markdown(
        "Please predict the Corrosion Efficiency of the Compound using the parameters!")
    st.error('Based on experiments, the Best Model is more recommended to predict the Compound well because it has a high accuracy value and a low error value. Other model combinations should only be used as a reference or perhaps just a comparison.')
    opsi_model = st.selectbox(
        "", ['Experimental Machine Learning','Analysis of the Best Model'])

    st.write("---")

    if opsi_model == "Experimental Machine Learning":
        base_machine_learning()
    else:
        analisis_model_terbaik()


# ========================= Ahkir halaman App ===================================================================================================================================================================================================================================================================================================================================================================

# Data Analisis
if menuweb == "Data Analysis":
    st.title("Dataset")
    analisis()
