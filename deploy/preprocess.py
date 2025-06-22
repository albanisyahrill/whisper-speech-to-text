import librosa
import noisereduce as nr
from peft import PeftConfig
from transformers import pipeline, WhisperForConditionalGeneration
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
)

language = 'indonesian'
task = 'transcribe'

peft_model_id = './Final-result/checkpoint-5245'
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map='auto')
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

whisper_asr = pipeline('automatic-speech-recognition', model=model, chunk_length_s=30, stride_length_s=10, batch_size=32, tokenizer=tokenizer, feature_extractor=feature_extractor, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'.replace('.', ',')

def process_audio_and_create_txt(audio_filename, whisper_asr, output_filename=None, noise=False):
    if noise:
        audio, sr = librosa.load(audio_filename, sr=16000)
        reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)
        prediction = whisper_asr(reduced_noise_audio, return_timestamps=True)

    prediction = whisper_asr(audio_filename, return_timestamps=True)

    txt_file_name = output_filename if output_filename else 'transcription.txt'

    txt_content = 'Hasil Transkripsi\n\n'
    for chunk in prediction['chunks']:
        start, end = chunk['timestamp']
        start_time = format_time(start)
        end_time = format_time(end)
        text = chunk['text']
        txt_content += f'{start_time} --> {end_time}\n{text}\n\n'
    
    with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
        txt_file.write(txt_content)

    return txt_content