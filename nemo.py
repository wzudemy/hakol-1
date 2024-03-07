import nemo.collections.asr as nemo_asr
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_small')
decision = speaker_model.verify_speakers('path/to/one/audio_file','path/to/other/audio_file')