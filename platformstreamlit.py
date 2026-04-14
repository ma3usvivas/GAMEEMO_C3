import streamlit as st
import requests

st.set_page_config(page_title="GAMEEMO Predictor", layout="wide")

st.title("Predição GAMEEMO por Subject")

st.write("Envie um arquivo .zip contendo os dados do subject (ex: S28.zip)")

uploaded_file = st.file_uploader("Upload do ZIP", type=["zip"])

if uploaded_file:

    with st.spinner("Processando..."):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict_subject",
                files={"file": uploaded_file}
            )

            result = response.json()

            st.success("Predição concluída!")
            
            for game, preds in result.items():

                st.subheader(f"🎮 {game}")

                for label, values in preds.items():
                    st.write(f"**{label}**: {values}")

        except Exception as e:
            st.error(f"Erro: {e}")