#!/bin/bash
#
# Run instruction library test overnight
# Tests all emotion variants to find the best instructions
#

set -e

cd /home/chris/projects/her

echo "========================================"
echo "Instruction Library Test"
echo "========================================"
echo "Time: $(date)"
echo ""

# Activate test venv
source test-venv/bin/activate

# Check TTS server
echo "Checking TTS server..."
curl -s http://localhost:8032/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Server: {d[\"model\"]}, Speaker: {d[\"speaker\"]}')"
echo ""

# Create output directory
mkdir -p /home/chris/spark-voice-pipeline/instruction_tests

echo "Starting test in background..."
echo "Log: /home/chris/projects/her/instruction_test.log"
echo ""

cd /home/chris/projects/her/nexus-voice
nohup python3 -u test_instruction_library.py > /home/chris/projects/her/instruction_test.log 2>&1 &
TEST_PID=$!

sleep 5

if ps -p $TEST_PID > /dev/null 2>&1; then
    echo "Test started successfully!"
    echo "PID: $TEST_PID"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f /home/chris/projects/her/instruction_test.log"
    echo ""
    echo "Results will be saved to:"
    echo "  /home/chris/spark-voice-pipeline/instruction_library_results.md"
    echo "  /home/chris/spark-voice-pipeline/instruction_tests/all_results.json"
else
    echo "Test failed to start. Check log for errors:"
    tail -30 /home/chris/projects/her/instruction_test.log
fi
