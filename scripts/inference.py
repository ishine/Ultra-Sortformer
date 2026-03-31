from nemo.collections.asr.models import SortformerEncLabelModel

# Load model from Hugging Face
diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_8spk_v1")
diar_model.eval()

# Streaming parameters (recommended for best performance)
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300

# Run diarization
predicted_segments = diar_model.diarize(audio=["data/multispeaker_session.wav"], batch_size=1)

for segment in predicted_segments[0]:
    print(segment)
