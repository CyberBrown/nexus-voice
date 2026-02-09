#!/usr/bin/env python3
"""Prepare NeMo manifest from training data."""
import json
import os
import wave

audio_dir = '/data/her/her_lines_24k'
output_manifest = '/data/her/train_nemo.json'

count = 0
with open('/data/her/train_24k.jsonl', 'r') as f_in, open(output_manifest, 'w') as f_out:
    for line in f_in:
        item = json.loads(line)
        filename = os.path.basename(item['audio'])
        audio_path = os.path.join(audio_dir, filename)
        
        if os.path.exists(audio_path):
            with wave.open(audio_path, 'rb') as wav:
                duration = wav.getnframes() / float(wav.getframerate())
            
            f_out.write(json.dumps({
                'audio_filepath': audio_path,
                'text': item['text'],
                'duration': duration
            }) + '\n')
            count += 1

print(f'Created manifest with {count} samples')
