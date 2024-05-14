from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.route('/run-python-script', methods=['POST'])
def run_script():
    try:
        # Path to the Python script you want to run
        script_path = 'repromodel_core/demotrainer.py'
        
        # Run the script and capture the output
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'output': result.stdout, 'error': None})
        else:
            return jsonify({'output': result.stdout, 'error': result.stderr}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/api/files', methods=['GET'])
def get_txt_files():
    try:
        relative_path = 'logs/Training_logs'  # Replace with your folder path
         # Get the path of the script itself
        script_path = os.path.abspath(__file__) 
        # Get the directory of the script
        script_dir = os.path.dirname(script_path)

        # Define the relative path of the trainer script
    
        absolute_path = os.path.join(script_dir, relative_path)
    


        files = os.listdir(absolute_path)
        txt_files = [file for file in files if file.endswith('.txt')]
        return jsonify(txt_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)