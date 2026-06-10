#!/bin/bash
set -e

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the local dev model
ollama serve &
sleep 5
ollama pull llama3.2

# Python venv + dependencies
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo "Setup complete. Run: .venv/bin/python app.py"
