
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
import json

# Normalisasi
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Import Library Linear dan Nonliner

# Linear
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#Non Linear 
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import NuSVR, SVR

#spliting ML
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score,make_scorer

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, make_scorer, accuracy_score
import matplotlib.pyplot as plt


import dataset.path_data as dth

df = dth.data_home()
df2 = dth.df2()
data_unduh = dth.data_unduh()

def base_machine_learning():
    opsi_experiment = st.selectbox(
        "Pilih Experiment yang ingin dilakukan", ['Model Kustomisasi', 'Latih Data Baru'])
    st.write("---")
    if opsi_experiment == 'Latih Data Baru' :
        train_new_data()

    else :
        kolom_filteralgoritma = {
            "Linear Regression": "Linear Regression",
            "Ridge" : "Ridge Regression",
            "Lasso": "Lasso",
            "Elastic Net": "Elastic Net",
            "XGB Regressor" : "XGB Regressor",
            "CatBoost Regressor" : "CatBoost Regressor",
            "Light GBM Regressor" : "Light GBM Regressor",
            "Gradient Boosting Regressor": "Gradient Boosting Regressor",
            "AdaBoost Regressor" : "AdaBoost Regressor",
            "Random Forest Regressor": "Random Forest Regressor",
            "SVR": "SVR",
            "NuSVR": "NuSVR"
        }

        to_filteralgoritma = st.selectbox("Masukkan Jenis Algoritma", options=list(
            kolom_filteralgoritma.keys()), format_func=lambda x: kolom_filteralgoritma[x])

        kolom_filternormalisasi = {
            'MinMaxScaler()': 'Min Max',
            'StandardScaler()': 'Standard',
            'RobustScaler()': 'Robust',
            'None': 'Tanpa Normalisasi',
        }

        to_filternormalisasi = st.selectbox("Masukkan Jenis Normalisasi", options=list(
            kolom_filternormalisasi.keys()), format_func=lambda x: kolom_filternormalisasi[x])

        kolom_filtersplit = {
            '0.2': 'HOCV 80:20',
            '0.3': 'HOCV 70:30',
            '0.4': 'HOCV 60:40',
        }

        to_filtersplit = st.selectbox("Masukkan Rasio Splitting Data", options=list(
            kolom_filtersplit.keys()), format_func=lambda x: kolom_filtersplit[x])

        namamodel = f"{to_filteralgoritma}_{to_filtersplit}_{to_filternormalisasi}"

        st.subheader(f"{to_filteralgoritma} {to_filtersplit} {to_filternormalisasi}")

        # Muat model
        try:
            model = joblib.load(f"model/model_biasa/{namamodel}.joblib")
        except Exception as e:
            st.warning(
                f"Gagal memuat {namamodel} dengan joblib. Menggunakan pickle sebagai alternatif. Kesalahan: {e}")
            with open(f'model/model_biasa/{namamodel}.pkl', 'rb') as f:
                model = pickle.load(f)

        X = df2.drop("IE EXP (%)",axis=1)
        X = X.astype(float)

        y = df2["IE EXP (%)"]
        y = y.astype(float)

        float_split = float(to_filtersplit)

        if to_filternormalisasi != "None":
            scaler = eval(to_filternormalisasi)
            x_normalisasi = scaler.fit_transform(X)
        else:
            x_normalisasi = X

        ## kfold cv
        # if float_split >= 3 :
        #     model.fit(x_normalisasi, y)

        #     kfold = KFold(n_splits=int(float_split), shuffle=True, random_state=42)

        #     # Define scoring metrics
        #     scoring = {'r2': make_scorer(r2_score), 'mae': make_scorer(mean_absolute_error), 'rmse': make_scorer(mean_squared_error, squared=False)}

        #     # Lakukan validasi
        #     r2_scores = cross_val_score(model, x_normalisasi, y, cv=kfold, scoring=scoring['r2'])
        #     rmse_scores = cross_val_score(model, x_normalisasi, y, cv=kfold, scoring=scoring['rmse'])
        #     mae_scores = cross_val_score(model, x_normalisasi, y, cv=kfold, scoring=scoring['mae'])

        #     rmse = np.mean(rmse_scores)
        #     r2 = np.mean(r2_scores)
        #     mae = np.mean(mae_scores)

        # else :
        X_train, X_test, y_train, y_test = train_test_split(
            x_normalisasi, y, test_size=float_split,random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)

        mae = mean_absolute_error(y_train,y_pred)
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_train, y_pred)

        # Display the mean squared error
        st.write(f"R2-Squared: ",r2," || Root Mean Squared Error: ", round(rmse, 6)," || Mean Absolute Error ",round(mae,6))


        tab4, tab5 = st.tabs(["Prediksi input data", "Prediksi input file"])

        with tab4:
            # Input manual
            col8, col9 = st.columns(2)

            with col8:
                molecular_weight = st.number_input(
                    "Masukkan Molecular Weight (g/mol): ", value=0.0)
                pKa = st.number_input("Masukkan pKa: ", value=0.0)
                log_P = st.number_input("Masukkan Log P: ", value=0.0)
                log_S = st.number_input("Masukkan Log S: ", value=0.0)
                polar_surface_area = st.number_input(
                    "Masukkan Polar Surface Area (Å2): ", value=0.0)
            with col9:
                polarizability = st.number_input(
                    "Masukkan Polarizability (Å3): ", value=0.0)
                HOMO = st.number_input("Masukkan HOMO (eV): ", value=0.0)
                LUMO = st.number_input("Masukkan LUMO (eV): ", value=0.0)
                electronegativity = st.number_input(
                    "Masukkan Electronegativity (eV): ", value=0.0)
                delta_N_Fe = st.number_input("Masukkan ΔN_Fe: ", value=0.0)

            with col8:
                y_new_pred = ""
                if st.button("Prediksi CIE%"):
                    if all([molecular_weight, pKa, log_P, log_S, polar_surface_area, polarizability, HOMO, LUMO, electronegativity, delta_N_Fe]):
                        x_user = pd.DataFrame({
                            'Molecular_weight MW (g/mol)': [molecular_weight],
                            'pKa': [pKa],
                            'Log P': [log_P],
                            'Log S': [log_S],
                            'Polar Surface Area (Å2)': [polar_surface_area],
                            'Polarizability (Å3)': [polarizability],
                            'HOMO (eV)': [HOMO],
                            'LUMO (eV)': [LUMO],
                            'Electronegativity (eV)': [electronegativity],
                            ' ΔN_Fe ': [delta_N_Fe]
                        })

                        # columns to scale
                        columns_to_scale = ['Molecular_weight MW (g/mol)', 'pKa', 'Log P', 'Log S', 'Polar Surface Area (Å2)',
                                            'Polarizability (Å3)', 'HOMO (eV)', 'LUMO (eV)', 'Electronegativity (eV)', ' ΔN_Fe ']

                        if to_filternormalisasi == "MinMaxScaler()":
                            try:
                                # Load scaler dari JSON file
                                with open('model/min_max_values.json', 'r') as file:
                                    scaler_params = json.load(file)

                                for feature in columns_to_scale:
                                    min_value = scaler_params[feature]["min"]
                                    max_value = scaler_params[feature]["max"]
                                    x_user[feature] = (
                                        x_user[feature] - min_value) / (max_value - min_value)
                            except Exception as e:
                                print(
                                    "Error when applying Min Max Scaling: ", str(e))

                        elif to_filternormalisasi == "StandardScaler()":
                            try:
                                # Load scaler dari JSON file
                                with open('model/standard_scaler_parameters.json', 'r') as file:
                                    scaler_params = json.load(file)

                                for feature in columns_to_scale:
                                    mean_value = scaler_params[feature]["mean"]
                                    std_value = scaler_params[feature]["std"]

                                    # Cek untuk menghindari pembagian dengan 0
                                    if std_value == 0:
                                        continue

                                    x_user[feature] = (
                                        x_user[feature] - mean_value) / std_value
                            except Exception as e:
                                print(
                                    "Error when applying Standard Scaling: ", str(e))

                        elif to_filternormalisasi == 'RobustScaler()':
                            try:
                                # Load scaler dari JSON file
                                with open('model/robust_scaler_parameters.json', 'r') as file:
                                    scaler_params = json.load(file)

                                # Set scaler parameters 
                                for feature in columns_to_scale:
                                    center_value = scaler_params[feature]["center"]
                                    scale_value = scaler_params[feature]["scale"]

                                    # Cek untuk menghindari pembagian dengan 0
                                    if scale_value == 0:
                                        continue

                                    x_user[feature] = (
                                        x_user[feature] - center_value) / scale_value
                            except Exception as e:
                                print(
                                    "Error when applying Robust Scaling: ", str(e))

                        y_new_pred = model.predict(x_user)

                        y_new_pred = str(y_new_pred[0].round(2))
                        st.subheader(
                            "Prediksi Efisiensi Korosi sebesar "+y_new_pred+" %")
                    else:
                        st.warning(
                            "Mohon isi semua nilai sebelum melakukan prediksi.")
        with tab5:
            # Upload CSV file
            uploaded_file = st.file_uploader("Upload file CSV untuk diprediksi", type=["csv"])

            if uploaded_file is not None:
                # Read CSV file
                df_uploaded = pd.read_csv(uploaded_file)
                df_uploaded = df_uploaded.drop(columns = 'IE EXP (%)')
                df_concat = df_uploaded.copy()
                df_concat = df_concat.astype(float)

                # Define columns to scale
                columns_to_scale = ['Molecular_weight MW (g/mol)', 'pKa', 'Log P', 'Log S', 'Polar Surface Area (Å2)',
                                    'Polarizability (Å3)', 'HOMO (eV)', 'LUMO (eV)', 'Electronegativity (eV)', ' ΔN_Fe ']

                if to_filternormalisasi == "MinMaxScaler()":
                    try:
                        # Load scaler dari JSON file
                        with open('model/min_max_values.json', 'r') as file:
                            normalization_params = json.load(file)

                        for feature in columns_to_scale:
                            min_value = normalization_params[feature]["min"]
                            max_value = normalization_params[feature]["max"]
                            df_uploaded[feature] = (
                                df_uploaded[feature] - min_value) / (max_value - min_value)
                    except Exception as e:
                        print("Error when applying Min Max Scaling: ", str(e))

                elif to_filternormalisasi == "StandardScaler()":
                    try:
                        # Load scaler dari JSON file
                        with open('model/standard_scaler_parameters.json', 'r') as file:
                            scaler_params = json.load(file)

                        for feature in columns_to_scale:
                            mean_value = scaler_params[feature]["mean"]
                            std_value = scaler_params[feature]["std"]

                            # Cek untuk menghindari pembagian dengan 0
                            if std_value == 0:
                                continue

                            df_uploaded[feature] = (
                                df_uploaded[feature] - mean_value) / std_value
                    except Exception as e:
                        print("Error when applying Standard Scaling: ", str(e))

                elif to_filternormalisasi == 'RobustScaler()':
                    try:
                        # Load scaler dari JSON file
                        with open('model/robust_scaler_parameters.json', 'r') as file:
                            scaler_params = json.load(file)

                        # Set scaler parameters 
                        for feature in columns_to_scale:
                            center_value = scaler_params[feature]["center"]
                            scale_value = scaler_params[feature]["scale"]

                            # Cek untuk menghindari pembagian dengan 0
                            if scale_value == 0:
                                continue

                            df_uploaded[feature] = (
                                df_uploaded[feature] - center_value) / scale_value
                    except Exception as e:
                        print("Error when applying Robust Scaling: ", str(e))

                # prediksi data
                predictions = model.predict(df_uploaded)

                st.write("Prediksi IE data yang diunggah: ")

                predictions_df = pd.DataFrame({namamodel: predictions})

                # Concatenate data prediksi dengan data inputan
                merged_df = pd.concat([df_concat, predictions_df], axis=1)
                st.dataframe(merged_df)

                st.download_button(
                    label="Download Predictions",
                    data=merged_df.to_csv(index=False),
                    file_name=f"predictions_{namamodel}.csv",
                    key="download_predictions",
                )

def analisis_model_terbaik():
    # Muat model
    try:
        model_untuk_analisis = joblib.load(f"model/best_model/new_version/modelcatboost_forweb.joblib")
    except Exception as e:
        st.warning(
            f"Gagal memuat model dengan joblib. Menggunakan pickle sebagai alternatif. Kesalahan: {e}")
        with open(f'model/best_model/new_version/modelcatboost_forweb.pkl', 'rb') as f:
            model_untuk_analisis = pickle.load(f)
    st.subheader("Gradient Boosting Regressor with Polynomial Regression")
    st.write(f"R2-Squared: ",float(0.9999)," || Root Mean Squared Error: ", float(0.0003))

    st.subheader("Scaling")
    st.write('Scaling data dilakukan untuk memastikan setiap fitur memiliki skala nilai yang konsisten, scaling dilakukan dengan menerapkan metode MinMax. MinMax Scaler adalah metode penskalaan yang mengubah ulang nilai-nilai setiap fitur dalam sebuah dataset ke dalam rentang yang umum. Metode ini melibatkan pengurangan nilai minimum dari setiap fitur dari semua nilai dalam fitur tersebut, kemudian membaginya dengan rentang nilai dalam fitur tersebut.')
    st.subheader("Splitting Data")
    st.write('Sebelum melakukan process modelling yang lebih lanjut, data displit menggunakan HOCV 60 : 40. Sehingga Data akan dibagi menjadi 2 jenis dengan partisi yang berbeda yaitu Data Training dengan partisi 60\% dan Data Testing 40/%.')
    st.subheader("Algoritma Machine Learning")
    st.write('Untuk melatih data menjadi model Machine Learning yang baik, Gradient Boosting Regressor (GBR) dalam machine learning digunakan untuk memodelkan dan memprediksi variabel target yang bersifat kontinu. Ini adalah algoritma ensemble yang kuat yang bekerja dengan cara menggabungkan beberapa model kecil yang disebut pohon keputusan (decision trees) menjadi model yang lebih kompleks.')
    st.subheader("Optimasi Model")
    st.write('Model perlu dioptimalkan dengan menggunakan Polynomial Regression. Polynomial Regression digunakan untuk meningkatkan kemampuan model regresi dalam menangkap pola yang lebih kompleks dalam data. Dengan menerapkan polinomial, model regresi dapat memperhitungkan hubungan yang tidak linier antara variabel independen dan variabel dependen. Hal ini memungkinkan model untuk menyesuaikan dengan lebih baik terhadap data yang memiliki pola yang melengkung atau tidak linear.')
    
    st.subheader("Evaluasi Performa Machine Learning")
    st.image('assets/graf1.png', use_column_width=True)
    st.image('assets/plot_prediksi.png', use_column_width=True)
    st.image('assets/graf 2.png', use_column_width=True)

def train_new_data():
    upload_new_trainingdata = st.file_uploader("Upload Training Data dalam bentuk CSV", type=["csv"])

    if upload_new_trainingdata is not None : 
        df_new_data_ft = pd.read_csv(upload_new_trainingdata)
        df_for_training = df_new_data_ft.copy()

        # Heatmap korelasi
        st.subheader("Matriks Korelasi")
        correlation_matrix = df_for_training.corr(numeric_only = True)

        fig_heatmap = px.imshow(correlation_matrix, labels=dict(color="Korelasi"),
                                x=correlation_matrix.columns, y=correlation_matrix.columns,
                                color_continuous_scale="viridis")  
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
            width=800,
            height=600,  
        )

        st.plotly_chart(fig_heatmap)

        kolom_df_training = df_for_training.columns.to_list()

        kolom_df_training.remove('IE EXP (%)')

        # Pilih prediktor
        kolom_training = st.multiselect("Pilih Deskriptor Model Machine Learning", kolom_df_training, placeholder = "Tentukan kolom yang menjadi fitur prediktor, Secara default menggunakan keseluruhan fitur")
        if len(kolom_training) > 0 : 
            kolom_training.append("IE EXP (%)")
            df_training_selected = df_for_training[kolom_training]
        else : 
            df_training_selected = df_for_training

        kolom_algoritma_training = {
            "Linear Regression": LinearRegression(),
            "Ridge" : Ridge(),
            "Lasso": Lasso(),
            "Elastic Net": ElasticNet(),
            "XGB Regressor" : XGBRegressor(),
            "CatBoost Regressor" : CatBoostRegressor(),
            "Light GBM Regressor" : LGBMRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor" : AdaBoostRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "NuSVR":  NuSVR()
        }

        to_filteralgoritma_training = st.selectbox("Masukkan Jenis Algoritma", options=list(
            kolom_algoritma_training.keys()))
        
        kolom_normalisasi_training = {
            'MinMax Scaler': "MinMaxScaler()",
            'Standard Scaler': 'StandardScaler()',
            'Robust Scaler': 'RobustScaler()',
            'None': 'Tanpa Normalisasi',
        }

        to_filternormalisasi_training = st.selectbox("Masukkan Jenis Normalisasi", options=list(
            kolom_normalisasi_training.keys()))

        kolom_split_training = {
            '0.2': 'HOCV 80:20',
            '0.3': 'HOCV 70:30',
            '0.4': 'HOCV 60:40',
        }

        to_filtersplit_training = st.selectbox("Masukkan Rasio Splitting Data", options=list(
            kolom_split_training.keys()), format_func=lambda x: kolom_split_training[x])
        
        X = df_training_selected.drop("IE EXP (%)",axis=1)
        X = X.astype(float)

        y = df_for_training["IE EXP (%)"]
        y = y.astype(float)

        float_split = float(to_filtersplit_training)

        if kolom_normalisasi_training[to_filternormalisasi_training] != "Tanpa Normalisasi":
            scaler = eval(kolom_normalisasi_training[to_filternormalisasi_training])
            x_normalisasi = scaler.fit_transform(X)
        else:
            x_normalisasi = X

        model_new_training = kolom_algoritma_training[to_filteralgoritma_training]

        ## kfold cv
        # if float_split >= 3 :
        #     model_new_training.fit(x_normalisasi, y)

        #     kfold = KFold(n_splits=int(float_split), shuffle=True, random_state=42)

        #     # Define scoring metrics
        #     scoring = {'r2': make_scorer(r2_score), 'mae': make_scorer(mean_absolute_error), 'rmse': make_scorer(mean_squared_error, squared=False)}

        #     # Lakukan validasi silang
        #     r2_scores = cross_val_score(model_new_training, x_normalisasi, y, cv=kfold, scoring=scoring['r2'])
        #     rmse_scores = cross_val_score(model_new_training, x_normalisasi, y, cv=kfold, scoring=scoring['rmse'])
        #     mae_scores = cross_val_score(model_new_training, x_normalisasi, y, cv=kfold, scoring=scoring['mae'])

        #     rmse_training = np.mean(rmse_scores)
        #     r2_training = np.mean(r2_scores)
        #     mae_training = np.mean(mae_scores)

        #     # Display the mean squared error
        #     st.write(f"R2-Squared: ",r2_training," || Root Mean Squared Error: ", round(rmse_training,6)," || Mean Absolute Error ",round(mae_training,6))
            
        #     fig, ax = plt.subplots(figsize=(8, 6))
        #     ax.plot(np.arange(1, kfold.n_splits + 1), r2_scores, marker='o')
        #     ax.set_xlabel('Fold')
        #     ax.set_ylabel('Accuracy')
        #     ax.set_title('K-Fold Cross Validation Performance')
        #     ax.set_ylim(0, 1)
        #     ax.grid(True)

        #     # Menampilkan plot di aplikasi Streamlit
        #     st.pyplot(fig)

        # else :
        X_train, X_test, y_train, y_test = train_test_split(
            x_normalisasi, y, test_size=float_split,random_state=42)

        model_new_training.fit(X_train, y_train)

        y_pred = model_new_training.predict(X_train)

        mae_training = mean_absolute_error(y_train,y_pred)
        mse_training = mean_squared_error(y_train, y_pred)
        rmse_training = np.sqrt(mse_training)
        r2_training = r2_score(y_train, y_pred)

        # Display the mean squared error
        st.write(f"R2-Squared: ",r2_training," || Root Mean Squared Error: ", round(rmse_training,6)," || Mean Absolute Error ",round(mae_training,6))
        fig, ax = plt.subplots()
        ax.scatter(y_train, y_pred, c='b', label='Data Point')
        ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='m', linestyle='dotted', label='Garis Prediksi')
        ax.set_xlabel("Nilai Aktual")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title("Nilai Prediksi dan Aktual")
        ax.legend()

        st.pyplot(fig)

# def model_terbaik() :
#     kolom_filteralgoritma_best = {
#         "modelgbrpoly_forweb": "Gradient Boosting Polynomial 60 : 40",
#         "modelcatboost_forweb": "Cat Boost Polynomial 80 : 20",
#         "modelrandomforest_forweb": "Random Forest Regressor 70 : 30",
#     }

#     to_filteralgoritma_best = st.selectbox("Masukkan Jenis Model", options=list(
#         kolom_filteralgoritma_best.keys()), format_func=lambda x: kolom_filteralgoritma_best[x])

#     namamodel_best = f"{to_filteralgoritma_best}"
    
#     # Muat model
#     try:
#         with open(f'model/best_model/{namamodel_best}.pkl', 'rb') as f:
#             model_best = pickle.load(f)
#     except:
#         with open(f'model/best_model/{namamodel_best}.joblib', 'rb') as f:
#             model_best = joblib.load(f)
    
#     if namamodel_best == 'modelgbrpoly_forweb' :
#         test_split = 0.4
#     elif namamodel_best == 'modelcatboost_forweb' :
#         test_split = 0.2
#     elif namamodel_best == 'modelrandomforest_forweb' :
#         test_split = 0.3

#     X = df2.drop("IE EXP (%)",axis=1)
#     X = X.astype(float)

#     y = df2["IE EXP (%)"]
#     y = y.astype(float)

#     scaler = eval("MinMaxScaler()")
#     x_normalisasi = scaler.fit_transform(X)
#     y_normalisasi = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

#     X_train, X_test, y_train, y_test = train_test_split(
#             x_normalisasi, y_normalisasi, test_size=test_split, random_state=42)

#     model_best.fit(X_train, y_train)

#     y_pred = model_best.predict(X_train)

#     mae = mean_absolute_error(y_train,y_pred)
#     mse = mean_squared_error(y_train, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_train, y_pred)*100

#     # Display the mean squared error
#     st.write(f"R2-Score: ",round(r2,2)," %"," || Root Mean Squared Error: ", round(rmse, 2)," || Mean Absolute Error ",round(mae,2))

#     tab4_best, tab5_best = st.tabs(
#         ["Prediksi input data", "Prediksi input file"])

#     with tab4_best:
#         # Input manual
#         col8_best, col9_best = st.columns(2)

#         with col8_best:
#             molecular_weight_best_1 = st.number_input(
#                 "Masukkan Molecular Weight (g/mol): ", value=0.0)
#             pKa_best_1 = st.number_input("Masukkan pKa: ", value=0.0)
#             log_P_best_1 = st.number_input("Masukkan Log P: ", value=0.0)
#             log_S_best_1 = st.number_input("Masukkan Log S: ", value=0.0)
#             polar_surface_area_best_1 = st.number_input(
#                 "Masukkan Polar Surface Area (Å2): ", value=0.0)
#         with col9_best:
#             polarizability_best_1 = st.number_input(
#                 "Masukkan Polarizability (Å3): ", value=0.0)
#             HOMO_best_1 = st.number_input(
#                 "Masukkan HOMO (eV): ", value=0.0)
#             LUMO_best_1 = st.number_input(
#                 "Masukkan LUMO (eV): ", value=0.0)
#             electronegativity_best_1 = st.number_input(
#                 "Masukkan Electronegativity (eV): ", value=0.0)
#             delta_N_Fe_best_1 = st.number_input(
#                 "Masukkan ΔN_Fe: ", value=0.0)

#         with col8_best:
#             y_new_pred = ""
#             if st.button("Prediksi CIE%"):
#                 if all([molecular_weight_best_1, pKa_best_1, log_P_best_1, log_S_best_1, polar_surface_area_best_1, polarizability_best_1, HOMO_best_1, LUMO_best_1, electronegativity_best_1, delta_N_Fe_best_1]):
#                     x_user_best = pd.DataFrame({
#                         'Molecular_weight MW (g/mol)': [molecular_weight_best_1],
#                         'pKa': [pKa_best_1],
#                         'Log P': [log_P_best_1],
#                         'Log S': [log_S_best_1],
#                         'Polar Surface Area (Å2)': [polar_surface_area_best_1],
#                         'Polarizability (Å3)': [polarizability_best_1],
#                         'HOMO (eV)': [HOMO_best_1],
#                         'LUMO (eV)': [LUMO_best_1],
#                         'Electronegativity (eV)': [electronegativity_best_1],
#                         ' ΔN_Fe ': [delta_N_Fe_best_1]
#                     })

#                     # Define columns to scale
#                     columns_to_scale_best_1 = ['Molecular_weight MW (g/mol)', 'pKa', 'Log P', 'Log S', 'Polar Surface Area (Å2)',
#                                                 'Polarizability (Å3)', 'HOMO (eV)', 'LUMO (eV)', 'Electronegativity (eV)', ' ΔN_Fe ']

#                     # Scale specific columns using normalization parameters
#                     try:
#                         # Load normalization parameters from JSON
#                         with open('model/min_max_values.json', 'r') as file:
#                             normalization_params_best = json.load(file)

#                         for feature in columns_to_scale_best_1:
#                             min_value = normalization_params_best[feature]["min"]
#                             max_value = normalization_params_best[feature]["max"]
#                             x_user_best[feature] = (
#                                 x_user_best[feature] - min_value) / (max_value - min_value)
#                     except Exception as e:
#                         print("Error when applying Min Max Scaling: ", str(e))

#                     y_new_pred_best = model_best.predict(x_user_best)

#                     X_max = 99.00
#                     X_min = 67.70
#                     y_new_pred_best = (
#                         y_new_pred_best * (X_max - X_min)) + X_min
#                     y_new_pred_best = str(y_new_pred_best[0].round(2))
#                     st.subheader(
#                         "Prediksi Efisiensi Korosi sebesar "+y_new_pred_best+" %")
#                 else:
#                     st.warning(
#                         "Mohon isi semua nilai sebelum melakukan prediksi.")
#     with tab5_best:
#         # Upload CSV file
#         uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#         if uploaded_file is not None:
#             # Read the uploaded CSV file
#             df_uploaded = pd.read_csv(uploaded_file)
#             df_concat = df_uploaded.copy()

#             # Define columns to scale
#             columns_to_scale = ['Molecular_weight MW (g/mol)', 'pKa', 'Log P', 'Log S', 'Polar Surface Area (Å2)',
#                                 'Polarizability (Å3)', 'HOMO (eV)', 'LUMO (eV)', 'Electronegativity (eV)', ' ΔN_Fe ']

#             # Scale specific columns using normalization parameters
#             try:
#                 # Load normalization parameters from JSON
#                 with open('model/min_max_values.json', 'r') as file:
#                     normalization_params = json.load(file)

#                 for feature in columns_to_scale:
#                     min_value = normalization_params[feature]["min"]
#                     max_value = normalization_params[feature]["max"]
#                     df_uploaded[feature] = (
#                         df_uploaded[feature] - min_value) / (max_value - min_value)
#             except Exception as e:
#                 print("Error when applying Min Max Scaling: ", str(e))

#             # Make predictions for each row in the uploaded file
#             predictions = model_best.predict(df_uploaded)

#             # Invert the predictions using your formula
#             X_max = 99.00
#             X_min = 68
#             predictions_invert = (predictions * (X_max - X_min)) + X_min

#             # Display the predictions
#             st.write("Predictions for the uploaded file:")

#             predictions_df = pd.DataFrame(
#                 {namamodel_best: predictions_invert})

#             # Concatenate the predictions DataFrame with the uploaded data
#             merged_df = pd.concat([df_concat, predictions_df], axis=1)

#             st.dataframe(merged_df)

#             # Download the merged DataFrame as a CSV file
#             st.download_button(
#                 label="Download Predictions",
#                 data=merged_df.to_csv(index=False),
#                 file_name="predictions.csv",
#                 key="download_predictions",
#             )

