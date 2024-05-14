from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json


app = Flask(__name__)
CORS(app)

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
        relative_path = 'logs'  # Replace with your folder path
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
    
@app.route('/submit-config-start-training', methods=['POST'])
def submit_config_start_training_():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            app.logger.error("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert the JSON data to a string to pass as an argument
        json_data = json.dumps(data)
        app.logger.info(f"JSON data to be passed to the script")
        
        # Path to the Python script you want to run
        script_path = 'repromodel_core/trainer.py'
        
        # Run the script and capture the output
        result = subprocess.run(
            ['python', script_path, json_data],
            capture_output=True,
            text=True
        )
        
        # Check subprocess result
        if result.returncode == 0:
            app.logger.info("Script executed successfully")
            return jsonify({'output': result.stdout, 'error': None})
        else:
            app.logger.error(f"Script execution failed: {result.stderr}")
            return jsonify({'output': result.stdout, 'error': result.stderr}), 400

    except Exception as e:
        app.logger.exception("An error occurred during the process")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005,threaded=True, debug=True)