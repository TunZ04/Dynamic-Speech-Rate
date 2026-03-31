"""
Use this file to change the rate of an audio file across the board. This step is done after the dynamic rate change.
You can modify the desired word rate and original word rate.
"""


import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
import soundfile as sf

desired_word_rate = 310
original_word_rate = 177

filename = "what-happened-when-i-started-measuring-my-life-every-day"
input_path = f"Input/{filename}.wav"

audio_data, sample_rate = sf.read(input_path)

speed_increase = desired_word_rate / original_word_rate
output_path = f"Output/{filename}_{speed_increase:.2f}x.wav"

reader = ArrayReader(audio_data.T)
writer = ArrayWriter(audio_data.shape[1])

tsm = wsola(reader.channels, speed=speed_increase)
tsm.run(reader, writer)

sf.write(output_path, writer.data.T, sample_rate)
print(f"New audio written to file")

