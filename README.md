# Speech-to-Text dengan Whisper + Streamlit Deploy

## Deskripsi Proyek

Notebook ini membangun pipeline transkripsi audio ke teks menggunakan model Whisper dari OpenAI. Folder `deploy/` berisi aplikasi Streamlit untuk menguji dan menggunakan model secara interaktif.

## Struktur Proyek
```
├── deploy/ # Folder deployment local
│ ├── app.py # Aplikasi streamlit
│ └── preprocess.py / # Modul untuk melakukan preprocessing input audio
├── .gitignore # File untuk membuat list file dan folder untuk diabaikan
├── Whisper_ASR.ipynb # Notebook utama pembuatan dan uji model Whisper
```

## Instalasi
1. Clone repositori:
```bash
git clone <repo-url>
cd <repo-folder>
```

2. Buat dan aktifkan virtual environment

```bash
python -m venv venv
venv/bin/activate
```

3. Instal dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan Notebook

Jalankan `notebook.ipynb` untuk melakukan:

- Load model Whisper
- Preprocessing dataset audio
- Training model
- Evaluasi model
- Simpan hasil model

## Menjalankan Streamlit

Pindah ke folder `deploy/` lalu jalankan:

```bash
streamlit run app.py
```

Aplikasi akan menyediakan UI untuk mengunggah audio dan menampilkan hasil transkripsi.

## Catatan Penting

- Pastikan file audio dalam format yang didukung Whisper (MP3 dan WAV).
- Model Whisper memerlukan `ffmpeg`, pastikan sudah terinstal di sistem Anda.

---

Lisensi: MIT

```

```
