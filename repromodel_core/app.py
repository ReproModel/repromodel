from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.route('/run-python-script', methods=['POST'])
def run_python_script():
    try:
        # Path to your Python script
        script_path = '/Users/julien/Documents/1_Repos/1_Private/repromodel/repromodel_core/demotrainer.py'
        
        # Run the script using subprocess
        result = subprocess.run(['python', script_path], capture_output=True, text=True, check=True)
        
        # Return the script's output
        return jsonify({
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            'stdout': e.stdout,
            'stderr': e.stderr,
            'returncode': e.returncode,
            'error': str(e)
        }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
folder_path = '/Users/julien/Documents/1_Repos/1_Private/repromodel/logs/Training_logs'  # Replace with your folder path
@app.route('/api/files', methods=['GET'])
def get_txt_files():
    try:
        files = os.listdir(folder_path)
        txt_files = [file for file in files if file.endswith('.txt')]
        return jsonify(txt_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(threaded=True)