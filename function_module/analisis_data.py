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

        st.title("Description of Chemical Features")
        # Deskripsi untuk setiap fitur
        features_descriptions = {
            "Molecular Weight (MW)": "The molecular mass of a compound, measured in atomic mass units (daltons or g/mol).",
            "Acid Dissociation Constant (pKa)": "A measure of the acidic or basic strength of a compound, indicating how easily it releases or accepts protons.",
            "Octanol-Water Partition Coefficient (log P)": "The ratio of a compound's concentration between octanol and water, indicating the relative solubility of the compound in the nonpolar and polar phases.",
            "Water Solubility (log S)": "The base 10 logarithm of the solubility of a compound in water.",
            "Polar Surface Area (PSA)": "The surface area of a molecule that can interact with polar solvents or hydrogen bonds.",
            "Polarizability (α)": "The ability of a molecule to undergo changes in electron distribution when in an electric field.",
            "Energy of Highest Occupied Molecular Orbital (E-HOMO)": "Energy of the top level of the occupied molecular orbital.",
            "Energy of Lowest Unoccupied Molecular Orbital (E-LUMO)": "Energy of the lowest level of unoccupied molecular orbitals.",
            "Electrophilicity (ω)": "A measure of how strongly a compound is able to attract electron pairs.",
            "The Fraction Electron Shared (∆N)": "The fraction of the number of electron pairs shared in a bond."
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

            if num_lines == 1:
                target_col.write("")
            else:
                target_col.markdown("---")

    with tabkorelasi:
        # Heatmap korelasi
        st.write("Heatmap of Correlation Between Variables")
        correlation_matrix = data_model_analisis.corr()

        fig_heatmap = px.imshow(correlation_matrix, labels=dict(color="Correlation"),
                                x=correlation_matrix.columns, y=correlation_matrix.columns,
                                color_continuous_scale="viridis")  
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                fig_heatmap.add_annotation(
                    x=correlation_matrix.columns[i],
                    y=correlation_matrix.columns[j],
                    text=f"{correlation_matrix.iloc[j, i]:.2f}",
                    showarrow=False,
                    font=dict(size=8)
                )
        
        fig_heatmap.update_layout(
            width=800,
            height=600,  
        )
        st.plotly_chart(fig_heatmap)

        # Scatter plot untuk dua variabel tertentu
        st.write("Scatter Plot for Two Specified Variables")
        selected_x_variable = st.selectbox("Variable X:", data_model_analisis.columns)
        selected_y_variable = st.selectbox("Variable Y:", data_model_analisis.columns)

        fig_scatter = px.scatter(data_model_analisis,
                                 x=selected_x_variable,
                                 y=selected_y_variable,
                                 labels={selected_x_variable: selected_x_variable,
                                         selected_y_variable: selected_y_variable},
                                 color_discrete_map={selected_x_variable: 'red', selected_y_variable: 'blue'},
                                 symbol_map={selected_x_variable: 'circle'})
        st.plotly_chart(fig_scatter)


    with tabprediksidata:
        st.subheader("Comparison of Model Prediction Results with Experimental Predictions")

        df_prediksi_data = dth.dataframe_prediksi_data()
        st.dataframe(df_prediksi_data)

        # Membuat line chart dengan data asli
        # kolom_filter_model = {
        #     'GBR 60:40 Polynomial': 'IE EXP (%)GBR_poli 6040 Polynomial 2',
        #     'CatBoost 80:20 Polynomial': 'modelcatbost_poly',
        #     'Random Forest 70:30': 'model_rf',
        # }

        kolom_filter_model = {
            'GBR 60:40 Polynomial': 'GBR 60:40 Polynomial',
            'CatBoost 80:20 Polynomial': 'CatBoost 80:20 Polynomial',
            'Random Forest 70:30': 'Random Forest 70:30',
        }

        to_filter_columns = st.selectbox("Enter Compound filter to display", options=list(
            kolom_filter_model.keys()), format_func=lambda x: kolom_filter_model[x])
        
        if to_filter_columns=="GBR 60:40 Polynomial":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)'],
                                    mode='lines+markers', name='Data Asli'))

            # line chart untuk model prediksi
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)GBR_poli 6040 Polynomial 2'],
                                    mode='lines+markers', name='Prediksi Model'))

            fig.update_layout(title='Original Data Comparison with GBR 60:40 Polynomial',
                            xaxis_title='Urutan Data',
                            yaxis_title='IE EXP (%)')
            
        elif to_filter_columns=="CatBoost 80:20 Polynomial":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)'],
                                    mode='lines+markers', name='Data Asli'))

            # line chart untuk model prediksi
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['modelcatbost_poly'],
                                    mode='lines+markers', name='Prediksi Model'))

            fig.update_layout(title='Comparison of Original Data with CatBoost 80:20 Polynomial',
                            xaxis_title='Data',
                            yaxis_title='IE EXP (%)')
        
        elif to_filter_columns=="Random Forest 70:30":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['IE EXP (%)'],
                                    mode='lines+markers', name='Data Asli'))

            fig.add_trace(go.Scatter(x=df_prediksi_data.index, y=df_prediksi_data['model_rf'],
                                    mode='lines+markers', name='Prediksi Model'))

            fig.update_layout(title='Comparison of Original Data with Random Forest 70:30',
                            xaxis_title='Data',
                            yaxis_title='IE EXP (%)') 
        
        fig.update_layout(
            height=750,  
            width=950    
        )

        st.plotly_chart(fig)
