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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, make_scorer

import dataset.path_data as dth

df = dth.data_home()
df2 = dth.df2()
data_unduh = dth.data_unduh()

def analisis() :
    tab_data,tabkorelasi,tabprediksidata = st.tabs(["DataSet","Korelasi","Hasil Prediksi Pengujian"])

    with tab_data:
        # Menampilkan DataFrame
        data_model_analisis = df[['Molecular_weight MW (g/mol)', 'pKa', 'Log P', 'Log S', 'Polar Surface Area',
                                  'Polarizability', 'HOMO (eV)', 'LUMO (eV)', 'Electronegativity (eV)', ' ?N_Fe ',
                                  'IE EXP (%)']]
        data_model_analisis.dropna(inplace=True)
        data_model_analisis.reset_index(drop=True, inplace=True)
        st.write(data_model_analisis)

        st.title("Deskripsi Fitur Kimia")
        # Deskripsi untuk setiap fitur
        features_descriptions = {
            "Molecular Weight (MW)": "Massa molekul suatu senyawa, diukur dalam satuan massa atom (dalton atau g/mol).",
            "Acid Dissociation Constant (pKa)": "Ukuran kekuatan asam atau basa suatu senyawa, menunjukkan seberapa mudah senyawa tersebut melepaskan atau menerima proton.",
            "Octanol-Water Partition Coefficient (log P)": "Rasio konsentrasi suatu senyawa antara oktanol dan air, mengindikasikan kelarutan relatif senyawa dalam fase nonpolar dan polar.",
            "Water Solubility (log S)": "Logaritma basis 10 dari kelarutan suatu senyawa dalam air.",
            "Polar Surface Area (PSA)": "Luas permukaan molekul yang dapat berinteraksi dengan pelarut polar atau ikatan hidrogen.",
            "Polarizability (α)": "Kemampuan molekul untuk mengalami perubahan distribusi elektron saat berada dalam medan listrik.",
            "Energy of Highest Occupied Molecular Orbital (E-HOMO)": "Energi tingkat teratas orbital molekul yang diisi.",
            "Energy of Lowest Unoccupied Molecular Orbital (E-LUMO)": "Energi tingkat terbawah orbital molekul yang kosong.",
            "Electrophilicity (ω)": "Ukuran seberapa kuat suatu senyawa mampu menarik pasangan elektron.",
            "The Fraction Electron Shared (∆N)": "Fraksi jumlah pasangan elektron yang dibagikan dalam suatu ikatan."
        }

        # Membagi deskripsi menjadi dua kolom
        col1, col2 = st.columns(2)

        # Menampilkan deskripsi untuk setiap fitur pada kolom masing-masing
        for i, (feature, description) in enumerate(features_descriptions.items()):
            lines = description.split("\n")
            num_lines = len(lines)

            # Menentukan kolom target berdasarkan indeks
            target_col = col1 if i % 2 == 0 else col2

            target_col.subheader(feature)
            target_col.write(description)

            # Tambahkan spasi hanya jika deskripsi memiliki satu baris
            if num_lines == 1:
                target_col.write("")  # Menambahkan spasi antar fitur jika deskripsi satu baris
            else:
                target_col.markdown("---")  # Garis pemisah antar fitur di kolom target

    with tabkorelasi:
        # Heatmap korelasi
        st.write("Heatmap Korelasi Antar Variabel:")
        correlation_matrix = data_model_analisis.corr()

        # Plot Heatmap dengan Plotly Express
        fig_heatmap = px.imshow(correlation_matrix, labels=dict(color="Korelasi"),
                                x=correlation_matrix.columns, y=correlation_matrix.columns,
                                color_continuous_scale="viridis")  # Anda dapat mengganti "viridis" dengan skala warna lainnya
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                fig_heatmap.add_annotation(
                    x=correlation_matrix.columns[i],
                    y=correlation_matrix.columns[j],
                    text=f"{correlation_matrix.iloc[j, i]:.2f}",
                    showarrow=False,
                    font=dict(size=8, color="white")
                )
        
        fig_heatmap.update_layout(
            height=600,  # Ganti nilai ini sesuai kebutuhan
            width=800    # Ganti nilai ini sesuai kebutuhan
        )
        st.plotly_chart(fig_heatmap)

        # Scatter plot untuk dua variabel tertentu
        st.write("Scatter Plot untuk Dua Variabel Tertentu:")
        selected_x_variable = st.selectbox("Pilih Variabel X:", data_model_analisis.columns)
        selected_y_variable = st.selectbox("Pilih Variabel Y:", data_model_analisis.columns)

        # Plot Scatter Plot dengan Plotly Express
        fig_scatter = px.scatter(data_model_analisis,
                                 x=selected_x_variable,
                                 y=selected_y_variable,
                                 labels={selected_x_variable: selected_x_variable,
                                         selected_y_variable: selected_y_variable},
                                 color_discrete_map={selected_x_variable: 'red', selected_y_variable: 'blue'},
                                 symbol_map={selected_x_variable: 'circle'})
        st.plotly_chart(fig_scatter)


    with tabprediksidata:
        st.write("ini prediksi")

        df_prediksi_data = dth.dataframe_prediksi_data()
        st.dataframe(df_prediksi_data)

        # Membuat line chart dengan data asli
        kolom_filter_model = {
            'GBR 60:40 Polynomial': 'IE EXP (%)GBR_poli 6040 Polynomial 2',
            'CatBoost 80:20 Polynomial': 'modelcatbost_poly',
            'Random Forest 70:30': 'model_rf',
        }

        to_filter_columns = st.selectbox("Masukkan filter Senyawa untuk ditampilkan", options=list(
            kolom_filter_model.keys()), format_func=lambda x: kolom_filter_model[x])
        
        if to_filter_columns=="GBR 60:40 Polynomial":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)'],
                                    mode='lines+markers', name='Data Asli'))

            # Menambahkan line chart untuk model prediksi
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)GBR_poli 6040 Polynomial 2'],
                                    mode='lines+markers', name='Prediksi Model'))

            fig.update_layout(title='Perbandingan Data Asli dengan GBR 60:40 Polynomial',
                            xaxis_title='Urutan Data',
                            yaxis_title='IE EXP (%)')
            
        elif to_filter_columns=="CatBoost 80:20 Polynomial":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)'],
                                    mode='lines+markers', name='Data Asli'))

            # Menambahkan line chart untuk model prediksi
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['modelcatbost_poly'],
                                    mode='lines+markers', name='Prediksi Model'))

            fig.update_layout(title='Perbandingan Data Asli dengan CatBoost 80:20 Polynomial',
                            xaxis_title='Urutan Data',
                            yaxis_title='IE EXP (%)')
        
        elif to_filter_columns=="Random Forest 70:30":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)'],
                                    mode='lines+markers', name='Data Asli'))

            # Menambahkan line chart untuk model prediksi
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['model_rf'],
                                    mode='lines+markers', name='Prediksi Model'))

            fig.update_layout(title='Perbandingan Data Asli dengan Random Forest 70:30',
                            xaxis_title='Urutan Data',
                            yaxis_title='IE EXP (%)') 
        
        fig.update_layout(
            height=750,  
            width=950    
        )

        st.plotly_chart(fig)
