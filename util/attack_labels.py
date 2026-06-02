"""Mappings between attack names and merged evaluation labels."""

id_to_label = {
    0: "Mars5",
    1: "MeloTTS",
    2: "Metavoice 1B",
    3: "MMS TTS (DE)",
    4: "FastPitch",
    5: "Glow-TTS (IT Female)",
    6: "Griffin-Lim",
    7: "Bark",
    8: "Tacotron2 (variants)",
    9: "SpeedySpeech",
    10: "VITS (variants)",
    11: "XTTS / vixTTS",
    12: "No Attack",
}

# your model_to_id mapping
model_to_id = {
    "-": 12,  # No Attack

    # XTTS / vixTTS
    "tts_models-multilingual-multi-dataset-xtts_v1.1": 11,
    "tts_models-multilingual-multi-dataset-xtts_v2": 11,
    "vixTTS": 11,
    "A29-XTTS": 11,

    # Griffin-Lim
    "griffin_lim": 6,

    # Bark
    "suno-bark-small": 7,
    "suno-bark": 7,
    "tts_models-multilingual-multi-dataset-bark": 7,

    # Glow-TTS
    "tts_models-en-ljspeech-glow-tts": 5,
    "tts_models-fa-custom-glow-tts": 5,
    "tts_models-uk-mai-glow-tts": 5,
    "tts_models-it-mai_male-glow-tts": 5,
    "tts_models-it-mai_female-glow-tts": 5,
    "tts_models-tr-common-voice-glow-tts": 5,
    "A01-GlowTTS": 5,
    "glowtts": 5,

    # Tacotron2
    "tts_models-en-ljspeech-tacotron2-DDC_ph": 8,
    "tts_models-en-ljspeech-tacotron2-DDC": 8,
    "tts_models-en-ljspeech-tacotron2-DCA": 8,
    "tts_models-fr-mai-tacotron2-DDC": 8,
    "tts_models-de-thorsten-tacotron2-DDC": 8,
    "tts_models-de-thorsten-tacotron2-DCA": 8,
    "tts_models-zh-CN-baker-tacotron2-DDC-GST": 8,
    "tts_models-en-ek1-tacotron2": 8,
    "A11-Tacotron2": 8,

    # FastPitch
    "tts_models-en-ljspeech-fast_pitch": 4,
    "A07-FastPitch": 4,
    "fastpitch": 4,

    # SpeedySpeech
    "tts_models-en-ljspeech-speedy-speech": 9,

    # VITS
    "tts_models-en-ljspeech-vits--neon": 10,
    "tts_models-en-ljspeech-vits": 10,
    "tts_models-de-css10-vits-neon": 10,
    "tts_models-lt-cv-vits": 10,
    "tts_models-fr-css10-vits": 10,
    "tts_models-it-mai_male-vits": 10,
    "tts_models-it-mai_female-vits": 10,
    "tts_models-bg-cv-vits": 10,
    "tts_models-ga-cv-vits": 10,
    "tts_models-sl-cv-vits": 10,
    "tts_models-lv-cv-vits": 10,
    "tts_models-mt-cv-vits": 10,
    "tts_models-bn-custom-vits-male": 10,
    "tts_models-el-cv-vits": 10,
    "tts_models-hu-css10-vits": 10,
    "tts_models-pt-cv-vits": 10,
    "tts_models-ro-cv-vits": 10,
    "tts_models-sv-cv-vits": 10,
    "tts_models-de-thorsten-vits": 10,
    "tts_models-da-cv-vits": 10,
    "A08-VITS": 10,
    "vits": 10,

    # MMS TTS (Facebook multilingual)
    "facebook-mms-tts-deu": 3,
    "facebook-mms-tts-swe": 3,
    "facebook-mms-tts-hun": 3,
    "facebook-mms-tts-rus": 3,
    "facebook-mms-tts-fin": 3,
    "facebook-mms-tts-nld": 3,
    "facebook-mms-tts-fra": 3,

    # Mars5
    "Mars5": 0,

    # MeloTTS
    "MeloTTS": 1,

    # Metavoice 1B
    "Metavoice-1B": 2,
}
