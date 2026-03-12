#!/bin/bash
# Convert audio file to 16 kHz mono WAV format

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [output_file]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}.wav}"

ffmpeg -i "$INPUT" -ar 16000 -ac 1 "$OUTPUT"

echo "Converted: $INPUT -> $OUTPUT"
