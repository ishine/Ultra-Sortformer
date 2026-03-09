#!/usr/bin/env python3
"""Minimal test for Ultra Diar Streaming Sortformer. Uses audio in workspace/test/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

from nemo.collections.asr.models import SortformerEncLabelModel

# Paths
WORKSPACE = Path(__file__).parent.parent
AUDIO_DIR = WORKSPACE / "test"
AUDIO_FILES = sorted(AUDIO_DIR.glob("*.flac")) or sorted(AUDIO_DIR.glob("*.wav"))
audio_paths = [str(p) for p in AUDIO_FILES]

if not audio_paths:
    print("No audio in test/")
    sys.exit(1)

# Load model
diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0")
diar_model.eval()

# Streaming params
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300

# Run diarization
predicted_segments = diar_model.diarize(audio=audio_paths, batch_size=1)

for i, segments in enumerate(predicted_segments):
    print(f"\n--- {Path(audio_paths[i]).name} ---")
    for segment in segments:
        print(segment)
