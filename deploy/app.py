import streamlit as st
from preprocess import process_audio_and_create_txt, whisper_asr

# Judul Aplikasi
st.title("Speech-to-text")

# Upload file audio
uploaded_file = st.file_uploader("Silahkan upload file audio anda", type=["wav", "mp3"])

# Proses file audio jika diunggah
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    st.write(f"File yang diunggah: {uploaded_file.name}")
    
    audio = uploaded_file.read()
    txt_content = process_audio_and_create_txt(audio, whisper_asr)
    txt_file_name = "transcription.txt"
        
    st.download_button(
        label="Download Transcription File",
        data=txt_content,
        file_name=txt_file_name,
        mime="text/txt"
    )