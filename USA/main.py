import torch
print("CUDA available:", torch.cuda.is_available())


import whisper
import subprocess
import os

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Helper to extract and chunk a given hour
def extract_and_chunk(hour_idx, chunk_start=1, chunk_end=12):
	hour_start = (hour_idx - 1) * 3600
	audio_hour_path = os.path.join(output_dir, f"audio_hour{hour_idx}.mp3")
	subprocess.run([
		"ffmpeg", "-y", "-i", "audio.mp3", "-ss", str(hour_start), "-t", "3600", "-acodec", "copy", audio_hour_path
	])
	chunk_files = []
	for i in range(chunk_start, chunk_end + 1):
		start = (i - 1) * 300
		out_file = os.path.join(output_dir, f"audio_hour{hour_idx}_chunk_{i}.mp3")
		subprocess.run([
			"ffmpeg", "-y", "-i", audio_hour_path, "-ss", str(start), "-t", "300", "-acodec", "copy", out_file
		])
		chunk_files.append(out_file)
	return chunk_files

# Load the Whisper model
model = whisper.load_model("base")

import re
full_text = ""
chunk_infos = []

num_hours = 8
chunks_per_hour = 12

# Build chunk_infos for all hours and all chunks
chunk_infos = [(hour, chunk) for hour in range(1, num_hours + 1) for chunk in range(1, chunks_per_hour + 1)]

# Extract and chunk all hours
chunk_files = []
for hour in range(1, num_hours + 1):
	chunk_files += extract_and_chunk(hour, chunk_start=1, chunk_end=chunks_per_hour)

# Map chunk_infos to actual file paths
chunk_file_map = {}
for hour, chunk in chunk_infos:
	chunk_file_map[(hour, chunk)] = os.path.join(output_dir, f"audio_hour{hour}_chunk_{chunk}.mp3")

# Transcribe and search clues
for idx, (hour, chunk) in enumerate(chunk_infos):
	out_file = chunk_file_map[(hour, chunk)]
	print(f"Transcribing hour {hour}, chunk {chunk} ({idx+1}/{len(chunk_infos)})...")
	result = model.transcribe(out_file, language="en", fp16=False)
	chunk_text = result["text"]
	full_text += chunk_text + "\n"
	for line in chunk_text.splitlines():
		lower_line = line.lower()
		for trigger in ['keyword', 'letter']:
			if trigger in lower_line:
				print(f"Found clue in hour {hour}, chunk {chunk} (trigger: '{trigger}'): {line}")
				break
	print(f"Progress: {((idx+1)/len(chunk_infos))*100:.0f}% done.")

print(full_text)

# Clean up chunk files
for f in chunk_files:
	try:
		os.remove(f)
	except Exception:
		pass
