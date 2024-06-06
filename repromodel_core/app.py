from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import subprocess
import os
import json
import ollama


app = Flask(__name__)
CORS(app)

# FUNCTION: Get custom templates

# Define the base directory relative to the location of this script
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

# Define the mapping of parameters to file paths
file_paths = {
    'models': os.path.join(BASE_DIR, 'models/customModel.py'),
    'augmentations': os.path.join(BASE_DIR, 'augmentations/customAugmentation.py'),
    'datasets': os.path.join(BASE_DIR, 'datasets/customDataset.py'),
    'early_stopping': os.path.join(BASE_DIR, 'early_stopping/customEarlyStopping.py'),
    'losses': os.path.join(BASE_DIR, 'losses/customLoss.py'),
    'metrics': os.path.join(BASE_DIR, 'metrics/customMetrics.py'),
    'preprocessing': os.path.join(BASE_DIR, 'preprocessing/customPreprocessor.py'),
    'postprocessing': os.path.join(BASE_DIR, 'postprocessing/customPostprocessor.py')
}

    
@app.route('/get-custom-script-template', methods=['GET'])
def get_custom_templates():
     # Get the parameter from the request
    file_type = request.args.get('type')
    
    # Check if the parameter is valid and corresponds to a file path
    if file_type in file_paths:
        file_path = file_paths[file_type]
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Return the file content
            return send_file(file_path, as_attachment=True)
        else:
             return jsonify({'error': f'File not found for type: {file_type}'}), 404
    else:
        return jsonify({'error': 'Invalid file type parameter'}), 400


#FEATURE: Save the custom script files
# Define base directory for file storage
BASE_DIR = 'repromodel_core/src'
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

# Ensure base directory and type directories exist
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

for type_dir in TYPE_DIRS.values():
    path = os.path.join(BASE_DIR, type_dir)
    if not os.path.exists(path):
        os.makedirs(path)

@app.route('/save-custom-script', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    file_type = request.form.get('type')
    filename = request.form.get('filename')

    if not file or not file_type or not filename:
        return jsonify({'error': 'Missing file, type or filename'}), 400

    if file_type not in TYPE_DIRS:
        return jsonify({'error': 'Invalid type'}), 400

    # Create the full path for the file
    save_path = os.path.join(BASE_DIR, TYPE_DIRS[file_type], f'{filename}.py')

    try:
        # Save the file
        file.save(save_path)
        # When successfull rerun generate choices
        script_path = 'repromodel_core/generate_choices.py'
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        return jsonify({'message': 'File uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FUNCTION: Retrieve the names of the training output files

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
    
# FUNCTION: Route for starting the TRAINING process from frontend    
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

#FUNCTION: for starting the TESTING process from frontend
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

# FUNCTION: to start TensorBoard
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


# Generate LLM Methodology Section

def generate_methodology(config, additionalPrompt, voice, format):
    # Construct the prompt
    prompt = f"Act like an experienced scientist and write a methodology section of the research paper based on this config file {config} and additionally follow these rules {additionalPrompt}. Write it in {voice} voice and use {format} format."

    # Call the ollama chat function
    response = ollama.chat(model='llama2', messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])

    return response['message']['content']

@app.route('/generate_methodology', methods=['POST'])
def generate_methodology_route():
    # Get the variables from the request
    config = request.form.get('config')
    additionalPrompt = request.form.get('additionalPrompt')
    voice = request.form.get('voice')
    format = request.form.get('format')

    # Generate the methodology section
    result = generate_methodology(config, additionalPrompt, voice, format)
    
    # Return the result as plain text
    return Response(result, mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005,threaded=True, debug=True , use_reloader=True)