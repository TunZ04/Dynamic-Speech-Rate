"""
This script takes a list of word speeds and a list of timestamps for each word
and renders the audio using WSOLA.
The two required lists were created and stored in Cache/ by calculate_speed_deltas.py
"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
import soundfile as sf
import pickle


NAME = "what-happened-when-i-started-measuring-my-life-every-day"
transcript_path = f"Input/{NAME}.txt"
audio_path = f"Input/{NAME}.wav"
output_path = f"Output/{NAME}.wav"

with open(f"Cache/wordspeed_deltas{NAME}.pkl", "rb") as f:
  wordspeed_deltas = pickle.load(f)
with open(f"Cache/word_times{NAME}.pkl", "rb") as f:
  word_times = pickle.load(f)

audio_data, sample_rate = sf.read(audio_path)


word_count = len(word_times)
total_speaking_time_seconds = len(audio_data) / sample_rate
base_speaking_rate_minutes = word_count * (60 / total_speaking_time_seconds)
base_speed_increase = 280 / base_speaking_rate_minutes
dynamic_speed_multiplier = 50 / base_speaking_rate_minutes

print(f"Original WPM: {base_speaking_rate_minutes:.0f}")


output_segments = []
for i, speed in enumerate(wordspeed_deltas):

  word_start = int(word_times[i].time_start * sample_rate)
  word_end = int(word_times[i+1].time_start * sample_rate)

  segment = audio_data[word_start:word_end]

  reader = ArrayReader(segment.T)
  writer = ArrayWriter(segment.shape[1])

  tsm = wsola(reader.channels, speed=1+speed*dynamic_speed_multiplier)
  tsm.run(reader, writer)

  output_segments.append(writer.data.T)

output = np.concatenate(output_segments)
sf.write(output_path, output, sample_rate)
print(f"New audio written to file")

new_speaking_time_seconds = len(output) / sample_rate
new_speaking_rate_minutes = word_count * (60 / new_speaking_time_seconds)
print(f"New WPM: {new_speaking_rate_minutes:.0f}")