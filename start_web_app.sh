#!/bin/bash
echo "Starting LMFRNet Web App..."
echo "Opening browser..."
if which xdg-open > /dev/null; then
  xdg-open http://localhost:8080 &
elif which open > /dev/null; then
  open http://localhost:8080 &
fi
echo "Starting local server with correct headers..."
python3 server.py
