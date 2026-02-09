#!/bin/bash
#
# Set up test environment with all required packages
#

set -e

cd /home/chris/projects/her

echo "Setting up test environment..."

# Create venv with system packages (for torch)
if [ ! -d "test-venv" ]; then
    python3 -m venv --system-site-packages test-venv
fi

source test-venv/bin/activate

# Install analysis packages
pip install --upgrade pip -q
pip install librosa parselmouth-praat praat-parselmouth speechbrain soundfile requests numpy scipy -q 2>/dev/null || \
pip install librosa soundfile requests numpy scipy -q

echo "Environment ready!"
python3 -c "import librosa; print(f'librosa: {librosa.__version__}')" || echo "librosa not available"
python3 -c "import parselmouth; print('parselmouth: OK')" || echo "parselmouth not available"
python3 -c "import soundfile; print('soundfile: OK')"
