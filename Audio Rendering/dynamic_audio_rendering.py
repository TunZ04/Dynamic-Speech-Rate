"""
This script takes a list of word timestamps and speeds
and renders the audio using WSOLA.
The two required lists were created and stored in Cache/ by calculate_speed_deltas.py
Inspired by JentGent's WSOLA pitch-shift 
"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
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

og_waveform, sample_rate = sf.read(audio_path)

word_count = len(word_times)
total_speaking_time_seconds = len(og_waveform) / sample_rate
base_speaking_rate_minutes = word_count * (60 / total_speaking_time_seconds)
base_speed_increase = 310 / base_speaking_rate_minutes
dynamic_speed_range = 60 / base_speaking_rate_minutes

og_sample_count = len(og_waveform)
new_sample_count = int(np.ceil(og_sample_count / base_speed_increase))

print(f"Original WPM: {base_speaking_rate_minutes:.0f}")

# normalise wordspeed deltas to the dynamic range centered around 1 #
arr_min = np.min(wordspeed_deltas)
arr_max = np.max(wordspeed_deltas)
if arr_min != arr_max:
  normalized_wordspeed_deltas = (0.5 + (wordspeed_deltas - arr_min) / (arr_max - arr_min)) * dynamic_speed_range


# calculating how much the dynamic speed multiplier will affect the overall speed increase
sample_count = 0
for i in range(word_count - 1):
  sample_count += (word_times[i+1].time_start - word_times[i].time_start) * (normalized_wordspeed_deltas[i])
sample_count += ((og_sample_count / sample_rate) - word_times[-1].time_start) * (normalized_wordspeed_deltas[-1])
sample_count *= sample_rate
avg_speed_mul = sample_count / og_sample_count

speeds = np.array(list(zip(map(lambda x: x.time_start * sample_rate, word_times), map(lambda x: x * base_speed_increase/avg_speed_mul, normalized_wordspeed_deltas))))


og_sample_count = len(og_waveform)

# calculating the new sample count size
new_sample_count = 0
for i in range(len(speeds)):
  if i + 1 < len(speeds):
    new_sample_count += np.ceil( (speeds[i+1][0] - speeds[i][0]) / speeds[i][1] )
  else:
    new_sample_count += max(0, np.ceil( (og_sample_count - speeds[i][0]) / speeds[i][1] ))


# tuning settings
WIN_LEN = 2024
WIN_FUNC = np.hanning(WIN_LEN)
SEARCH_RANGE = 200
# ANLS_HOP_LEN is the distance between the start of two windows in original og_waveform.
# make sure this is no less than the search range
ANLS_HOP_LEN = 200

# sneakily reduce og_waveform to monotone and add some buffer
og_waveform = np.append(og_waveform.mean(axis=1).astype(np.float64), np.zeros(WIN_LEN, dtype=np.float64))
new_sample_count = int(new_sample_count + WIN_LEN)
new_waveform = np.zeros(new_sample_count, dtype=np.float64)
new_waveform_norm = np.zeros(new_sample_count, dtype=np.float64)
new_scales = np.zeros(new_sample_count)


curr_speed_idx = 0
curr_speed = 1.0
synth_i = 0
synth_hop_len = int(ANLS_HOP_LEN / speeds[0][1]) # set starting synthesis hop length
# loop through windows
for i in range(ANLS_HOP_LEN, og_sample_count - ANLS_HOP_LEN, ANLS_HOP_LEN):

  # update the synthesis hop length
  if i > speeds[curr_speed_idx][0]:
    curr_speed = speeds[curr_speed_idx][1]
    synth_hop_len = int(ANLS_HOP_LEN / curr_speed)
    curr_speed_idx = min(curr_speed_idx + 1, len(speeds)-1)

  # get the ith synthesis window starting position
  synth_i += synth_hop_len

  # search for best analysis window placement
  overlap_ammount = WIN_LEN - synth_hop_len # number of samples overlapped in synth window
  synth_overlap = new_waveform_norm[synth_i : synth_i + overlap_ammount] * WIN_FUNC[:overlap_ammount]
  max_ws = -np.inf # we want to find the lowest area between overlapping waveforms (max wave similarity)
  for j in range(i - SEARCH_RANGE, i + SEARCH_RANGE):
    ws = np.dot(synth_overlap, og_waveform[j : j + overlap_ammount])
    if ws > max_ws:
      max_ws = ws
      overlap_start = j
  window = og_waveform[overlap_start : overlap_start + WIN_LEN] * WIN_FUNC[:WIN_LEN]
  
  synth_i_end = synth_i + WIN_LEN

  window_waveform = new_waveform[synth_i : synth_i_end]
  window_waveform += window

  window_scales = new_scales[synth_i : synth_i_end]
  window_scales += WIN_FUNC[:WIN_LEN]
  
  new_waveform_norm[synth_i : synth_i_end] = window_waveform / np.where(window_scales == 0, 1, window_scales)

  max_sample = synth_i_end


new_speaking_time_seconds = new_sample_count / sample_rate
new_speaking_rate_minutes = word_count * (60 / new_speaking_time_seconds)
print(f"New WPM: {new_speaking_rate_minutes:.0f}")

sf.write(f"Output/wpm{new_speaking_rate_minutes:.0f}_{NAME}.wav", new_waveform_norm, sample_rate)
print(f"New audio written to file")