Running the Talking Machine

1. Clone the repository

git clone https://github.com/DuaneRose/theTalkingMachine.git
cd theTalkingMachine

2. Create and activate the Python environment

macOS

python3.11 -m venv .venv
source .venv/bin/activate

Windows PowerShell

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

3. Install Python dependencies

python -m pip install -r requirements.txt

4. Install and start the Ollama model

Make sure Ollama is installed and running, then download the model:

ollama pull qwen3.5:4b

The model name in backEnd/brain.py should be:

brain = "qwen3.5:4b"

5. Start the application

From the project root:

cd backEnd
python bt.py

The application requires the Ollama service to be running. The first run may also need an internet connection to download the Hugging Face turn-detection model.
