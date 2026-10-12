"""
Setup & Run Script — HousePriceAI
Generates data, trains the model, then starts the Flask server.
"""
import subprocess
import sys
import os

def install_deps():
    print("[Setup] Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'])

def train_model():
    print("[Setup] Training model...")
    import train
    train.train()

def run_server():
    print("[Setup] Starting Flask server at http://localhost:5000")
    os.system(f'{sys.executable} app.py')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    install_deps()
    train_model()
    run_server()
