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
    
#Route for starting the TRAINING process from frontend    
@app.route('/submit-config-start-training', methods=['POST'])
def submit_config_start_training_():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            error_message = "No data provided in request"
            app.logger.error(error_message)
            return jsonify({'error': error_message}), 400
        
        # Convert the JSON data to a string to pass as an argument
        json_data = json.dumps(data)
        app.logger.info("Received JSON data for processing.")
        
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
            app.logger.info("Script executed successfully with output: %s", result.stdout)
            return jsonify({'output': result.stdout, 'error': None})
        else:
            error_detail = f"Script execution failed with error: {result.stderr}"
            app.logger.error(error_detail)
            return jsonify({'output': result.stdout, 'error': error_detail}), 400

    except Exception as e:
        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)
        return jsonify({'error': error_message}), 500

#Route for starting the TESTING process from frontend
@app.route('/submit-config-start-testing', methods=['POST'])
def submit_config_start_testing_():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            error_message = "No data provided in request"
            app.logger.error(error_message)
            return jsonify({'error': error_message}), 400
        
        # Convert the JSON data to a string to pass as an argument
        json_data = json.dumps(data)
        app.logger.info("Received JSON data for processing.")
        
        # Path to the Python script you want to run
        script_path = 'repromodel_core/tester.py'
        
        # Run the script and capture the output
        result = subprocess.run(
            ['python', script_path, json_data],
            capture_output=True,
            text=True
        )
        
        # Check subprocess result
        if result.returncode == 0:
            app.logger.info("Script executed successfully with output: %s", result.stdout)
            return jsonify({'output': result.stdout, 'error': None})
        else:
            error_detail = f"Script execution failed with error: {result.stderr}"
            app.logger.error(error_detail)
            return jsonify({'output': result.stdout, 'error': error_detail}), 400

    except Exception as e:
        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)
        return jsonify({'error': error_message}), 500

# Function to start TensorBoard
def start_tensorboard(logdir="logs"):
    # Kill any existing TensorBoard instances
    subprocess.run(['pkill', '-f', 'tensorboard'])
    
    # Start a new TensorBoard instance
    command = ['tensorboard', '--logdir', logdir]
    tensorboard_proc = subprocess.Popen(command)
    
    return f"TensorBoard started at http://localhost:6006 with logdir {logdir}"

# Route to start TensorBoard
@app.route('/start-tensorboard')
def tensorboard():
    log_dir = "repromodel_core/logs"  # Customize this path to where your logs are
    message = start_tensorboard(log_dir)
    return jsonify({"message": message})


@app.route('/generate-dummy-data', methods=['GET'])
def generate_data():
    try:
        # Path to the Python script you want to run
        script_path = 'repromodel_core/data/dummyData/generate_data.py'
        
        # Run the script and capture the output
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'output': result.stdout, 'error': None})
        else:
            return jsonify({'output': result.stdout, 'error': result.stderr}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005,threaded=True, debug=True)