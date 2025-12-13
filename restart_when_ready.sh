#!/bin/bash
TARGET_PID=61897
echo "Waiting for training process (PID $TARGET_PID) to finish..."
while kill -0 $TARGET_PID 2>/dev/null; do 
    sleep 10
done
echo "Training finished. Restarting bot..."
# Ensure execution permissions 
chmod +x run_bot.sh
./run_bot.sh
