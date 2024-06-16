from flask import Flask, jsonify, Response, request, send_file
from flask_cors import CORS

import json
import subprocess
import ollama
import os



######################################################################
# CONSTANTS
######################################################################


# Absolute path for file app.py.
APP_FILE = os.path.abspath(__file__)

# Absolute path for directory repromodel_core/src.
BASE_DIR = os.path.join(os.path.dirname(APP_FILE), 'src')

# Mapping of file type to file path.
FILE_PATHS = {
    'augmentations': os.path.join(BASE_DIR, 'augmentations/customAugmentation.py'),
    'datasets': os.path.join(BASE_DIR, 'datasets/customDataset.py'),
    'early_stopping': os.path.join(BASE_DIR, 'early_stopping/customEarlyStopping.py'),
    'losses': os.path.join(BASE_DIR, 'losses/customLoss.py'),
    'metrics': os.path.join(BASE_DIR, 'metrics/customMetrics.py'),
    'models': os.path.join(BASE_DIR, 'models/customModel.py'),
    'postprocessing': os.path.join(BASE_DIR, 'postprocessing/customPostprocessor.py'),
    'preprocessing': os.path.join(BASE_DIR, 'preprocessing/customPreprocessor.py')
}

# Define base directory for file storage.
BASE_DIR = 'repromodel_core/src'

# Define type directory for file storage.
TYPE_DIRS = {
    'augmentations': 'augmentations',
    'datasets': 'datasets',
    'early_stopping': 'early_stopping',
    'losses': 'losses',
    'metrics': 'metrics',
    'models': 'models',
    'postprocessing': 'postprocessing',
    'preprocessing': 'preprocessing'

}



######################################################################
# INITIALIZATION
######################################################################


# Initialize Flask app.
app = Flask(__name__)
CORS(app)

# Ensure base directory and type directories exist
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

for type_dir in TYPE_DIRS.values():
    path = os.path.join(BASE_DIR, type_dir)
    if not os.path.exists(path):
        os.makedirs(path)



######################################################################
# API ENDPOINTS - HEADER
######################################################################


# GET /ping
# Description: Returns whether backend is up and running.
@app.route('/ping', methods=['GET'])
def ping():
    
    # Return HTTP 200 OK status code.
    return jsonify({"message": "pong"}), 200


# GET /generate-dummy-data
# Description: Run the script generate_data.py.
@app.route('/generate-dummy-data', methods=['GET'])
def generate_data():
    try:
        
        # Run the script generate_data.py and capture the output.
        command = ['python', 'repromodel_core/data/dummyData/generate_data.py']
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check the subprocess result.
        if result.returncode == 0:
            return jsonify({'output': result.stdout, 'error': None})
        
        else:
            # Return HTTP 400 Bad Request status code.
            return jsonify({'output': result.stdout, 'error': result.stderr}), 400
    
    except Exception as e:

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': str(e)}), 500



######################################################################
# API ENDPOINTS - Custom Script
######################################################################


# GET /get-custom-script-template
# Description: Get the custom script templates.
@app.route('/get-custom-script-template', methods=['GET'])
def get_custom_templates():
    
    # Get the 'type' parameter from the request.
    file_type = request.args.get('type')
    
    # Check if 'type' parameter has corresponding file path.
    if file_type in FILE_PATHS:
        file_path = FILE_PATHS[file_type]
        
        # Check if the script template file exists.
        if os.path.exists(file_path):
            
            # Return the file content.
            return send_file(file_path, as_attachment=True)
        
        else:
            # Return HTTP 404 Not Found status code.
             return jsonify({'error': f'File not found for type: {file_type}'}), 404
    
    else:
        # Return HTTP 400 Bad Request status code.
        return jsonify({'error': 'Invalid file type parameter.'}), 400


# POST /save-custom-script
# Description: Save the custom script file.
@app.route('/save-custom-script', methods=['POST'])
def save_custom_script():
    
    # Get the 'file' parameter from the request.
    file = request.files.get('file')

    # Get the 'type' parameter from the request.
    file_type = request.form.get('type')

    # Get the 'filename' parameter from the request.
    filename = request.form.get('filename')

    # Check for missing parameters.
    if not file or not file_type or not filename:

        # Return HTTP 400 Bad Request status code.
        return jsonify({'error': 'Missing file, type, or filename'}), 400

    # Check if file type is valid.
    if file_type not in TYPE_DIRS:

        # Return HTTP 400 Bad Request status code.
        return jsonify({'error': 'Invalid type.'}), 400

    # Absolute path for custom script file.
    save_path = os.path.join(BASE_DIR, TYPE_DIRS[file_type], f'{filename}.py')

    try:
        # Save the file.
        file.save(save_path)
        
        # Once file save is successful, rerun the script generate_choices.py.
        command = ['python', 'repromodel_core/generate_choices.py']
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Return HTTP 200 OK status code.
        return jsonify({'message': 'File uploaded successfully.'}), 200
    
    except Exception as e:

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': str(e)}), 500



######################################################################
# API ENDPOINTS - Experiment Builder
######################################################################


# POST /submit-config-start-training
# Description: Start the training process from frontend.    
@app.route('/submit-config-start-training', methods=['POST'])
def submit_config_start_training_():
    try:
        
        # Get JSON data from the request.
        data = request.get_json()
        
        if not data:
            error_message = "No data provided in request."
            app.logger.error(error_message)

            # Return HTTP 400 Bad Request status code.
            return jsonify({'error': error_message}), 400
        
        # Convert the JSON data to a string to pass as an argument.
        json_data = json.dumps(data)
        app.logger.info("Received JSON data for processing.")
        
        # Run the script trainer.py and capture the output.
        command = ['python', 'repromodel_core/trainer.py', json_data]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check the subprocess result.
        if result.returncode == 0:
            app.logger.info("Script executed successfully with output: %s", result.stdout)
            return jsonify({'output': result.stdout, 'error': None})
        
        else: 
            error_detail = f"Script execution failed with error: {result.stderr}"
            app.logger.error(error_detail)
            
            # Return HTTP 400 Bad Request status code.
            return jsonify({'output': result.stdout, 'error': error_detail}), 400

    except Exception as e:
        
        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': error_message}), 500



######################################################################
# API ENDPOINTS - Model Testing
######################################################################


# POST /submit-config-start-testing
# Description: Start the testing process from frontend.    
@app.route('/submit-config-start-testing', methods=['POST'])
def submit_config_start_testing_():
    try:
        
        # Get JSON data from the request.
        data = request.get_json()
        
        if not data:
            error_message = "No data provided in request."
            app.logger.error(error_message)

            # Return HTTP 400 Bad Request status code.
            return jsonify({'error': error_message}), 400
        
        # Convert the JSON data to a string to pass as an argument.
        json_data = json.dumps(data)
        app.logger.info("Received JSON data for processing.")
        
        # Run the script tester.py and capture the output.
        command = ['python', 'repromodel_core/tester.py', json_data]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check the subprocess result.
        if result.returncode == 0:
            app.logger.info("Script executed successfully with output: %s", result.stdout)
            return jsonify({'output': result.stdout, 'error': None})
        
        else:
            error_detail = f"Script execution failed with error: {result.stderr}"
            app.logger.error(error_detail)

            # Return HTTP 400 Bad Request status code.
            return jsonify({'output': result.stdout, 'error': error_detail}), 400

    except Exception as e:
        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': error_message}), 500



######################################################################
# API ENDPOINTS - Progress Viewer
######################################################################


# FUNCTION: Helper function to start TensorBoard.
def start_tensorboard(logdir="logs"):
    
    # Kill any existing TensorBoard instances.
    subprocess.run(['pkill', '-f', 'tensorboard'])
    
    # Start a new TensorBoard instance.
    command = ['tensorboard', '--logdir', logdir]
    tensorboard_proc = subprocess.Popen(command)
    
    return f"TensorBoard started at http://localhost:6006 with logdir {logdir}"


# GET /start-tensorboard
# Description: Start TensorBoard.
@app.route('/start-tensorboard')
def tensorboard():

    # Customize this path to where your logs are.
    log_dir = "repromodel_core/logs"

    message = start_tensorboard(log_dir)

    return jsonify({"message": message})


# GET /api/files
# Description: Retrieve the names of the training output files.
@app.route('/api/files', methods=['GET'])
def get_txt_files():
    try:
        
        # Replace with your folder path.
        relative_path = 'logs'
        
        # Get the path of the script itself.
        script_path = APP_FILE 
        
        # Get the directory of the script.
        script_dir = os.path.dirname(script_path)

        # Define the relative path of the trainer script.
        absolute_path = os.path.join(script_dir, relative_path)
    
        files = os.listdir(absolute_path)
        txt_files = [file for file in files if file.endswith('.txt')]
        
        return jsonify(txt_files)
    
    except Exception as e:

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({"error": str(e)}), 500



######################################################################
# API ENDPOINTS - LLM Description
######################################################################


# FUNCTION: Helper function to call LLM.
def generate_methodology(config, additionalPrompt, voice, format):
    
    # Construct the prompt to pass into LLM.
    prompt = f"Act like an experienced scientist and write a methodology section of the research paper based on this config file {config} and additionally follow these rules {additionalPrompt}. Write it in {voice} voice and use {format} format."

    # Call the ollama chat function.
    response = ollama.chat(model='llama3', messages=[
      {
        'role': 'user',
        'content': prompt,
      }
    ])

    # Return LLM response.
    return response['message']['content']


# GET /generate_methodology
# Description: Generate methodology section by calling LLM.
@app.route('/generate_methodology', methods=['POST'])
def generate_methodology_route():
    
    # Get the 'config' parameter from the request.
    config = request.form.get('config')

    # Get the 'additionalPrompt' parameter from the request.
    additionalPrompt = request.form.get('additionalPrompt')

    # Get the 'voice' parameter from the request.
    voice = request.form.get('voice')

    # Get the 'format' parameter from the request.
    format = request.form.get('format')

    # Generate the methodology section.
    result = generate_methodology(config, additionalPrompt, voice, format)
    
    # Return the result as plain text.
    return Response(result, mimetype='text/plain')



######################################################################
# MAIN METHOD
######################################################################


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005,threaded=True, debug=True , use_reloader=True)