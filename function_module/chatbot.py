import streamlit as st 
import json
import pandas as pd
import re  # Import library regular expression untuk manipulasi teks
import random  # Import library random untuk pemilihan respons acak

# Import fungsi load_model dari TensorFlow Keras
from tensorflow.keras.models import load_model
# Import Tokenizer untuk memproses teks
from tensorflow.keras.preprocessing.text import Tokenizer
# Import pad_sequences untuk padding urutan teks
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Import LabelEncoder untuk mengonversi label kelas
from sklearn.preprocessing import LabelEncoder

def chatbot():
    with open('model/intents copy.json', 'r', encoding="utf-8") as f:
        data = json.load(f)  # Membaca data dari file JSON

    # Membuat DataFrame dari data JSON
    df_chatbot = pd.DataFrame(data['intents'])

    # Membuat dictionary kosong untuk menyimpan data yang akan diubah ke DataFrame
    dic = {"tag": [], "patterns": [], "responses": []}
    for i in range(len(df_chatbot)):
        # Mengambil pola (patterns) dari DataFrame
        ptrns = df_chatbot[df_chatbot.index == i]['patterns'].values[0]
        # Mengambil respons dari DataFrame
        rspns = df_chatbot[df_chatbot.index == i]['responses'].values[0]
        # Mengambil tag dari DataFrame
        tag = df_chatbot[df_chatbot.index == i]['tag'].values[0]
        for j in range(len(ptrns)):
            dic['tag'].append(tag)  # Menambahkan tag ke dalam dictionary
            # Menambahkan pola ke dalam dictionary
            dic['patterns'].append(ptrns[j])
            # Menambahkan respons ke dalam dictionary
            dic['responses'].append(rspns)

    # Membuat DataFrame baru dari dictionary
    df_chatbot = pd.DataFrame.from_dict(dic)

    # Membuat objek Tokenizer dengan parameter tertentu
    tokenizer = Tokenizer(lower=True, split=' ')
    # Mengonversi teks pola menjadi urutan angka
    tokenizer.fit_on_texts(df_chatbot['patterns'])

    # Mengonversi teks pola menjadi urutan angka
    ptrn2seq = tokenizer.texts_to_sequences(df_chatbot['patterns'])
    # Melakukan padding terhadap urutan angka
    X = pad_sequences(ptrn2seq, padding='post')

    lbl_enc = LabelEncoder()  # Membuat objek LabelEncoder
    # Mengonversi label kelas menjadi angka
    y = lbl_enc.fit_transform(df_chatbot['tag'])

    # Memuat model yang telah dilatih sebelumnya
    model_path = 'model/my_model.keras'  # Perbarui dengan path yang benar
    loaded_model = load_model(model_path)  # Memuat model yang telah dilatih

    response_user = []  # List untuk menyimpan respons dari pengguna
    response_bot = []  # List untuk menyimpan respons dari bot
    robot_emoticon = "\U0001F916"
    # Menampilkan judul aplikasi chatbot
    st.title(f"{robot_emoticon} CorrosionShieldBot {robot_emoticon}")

    # menampilkan hasil histori dari chat sebelumnya
    if "messages" not in st.session_state:
        # Membuat session state untuk menyimpan histori chat sebelumnya
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Menampilkan histori chat sebelumnya
            st.markdown(message["content"])

    # Dapatkan input pengguna
    # Menampilkan input chat dan mendapatkan prompt dari pengguna
    prompt = st.chat_input("Ketik untuk memulai percakapan")

    # Proses input pengguna dan tampilkan respons
    if prompt:
        text = []
        # Menghapus karakter selain huruf dan tanda kutip dari prompt
        txt = re.sub('[^a-zA-Z\']', ' ', prompt)
        txt = txt.lower()  # Mengonversi prompt menjadi huruf kecil
        txt = txt.split()  # Membagi prompt menjadi kata-kata
        txt = " ".join(txt)  # Menggabungkan kata-kata kembali menjadi teks
        text.append(txt)  # Menambahkan teks ke dalam list

        # Mengonversi teks input pengguna menjadi urutan angka
        x_test = tokenizer.texts_to_sequences(text)
        # Melakukan padding terhadap urutan angka
        x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])
        # Memprediksi kelas dengan model yang telah dilatih
        y_pred = loaded_model.predict(x_test)
        y_pred = y_pred.argmax()  # Mengambil indeks kelas dengan nilai probabilitas tertinggi
        # Mengonversi indeks kelas kembali menjadi label kelas
        tag = lbl_enc.inverse_transform([y_pred])[0]
        # Mengambil respons berdasarkan label kelas
        responses = df_chatbot[df_chatbot['tag'] == tag]['responses'].values[0]

        # Gunakan respons tetap daripada random.choice(responses)
        # Memilih respons bot atau respons default jika tidak ada respons yang sesuai
        bot_response = random.choice(
            responses) if responses else "I cant understand what u say."

        with st.chat_message("user"):
            st.write(prompt)  # Menampilkan input pengguna
        with st.chat_message("assistant"):
            st.write(bot_response)  # Menampilkan respons bot
        # Menyimpan input pengguna ke dalam histori chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Menyimpan respons bot ke dalam histori chat
        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response})