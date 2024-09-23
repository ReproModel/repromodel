from flask import Flask, jsonify, Response, request, send_file
from flask_cors import CORS
from src.utils import copy_covered_files
import pandas as pd
import numpy as np
import sys
import os
from io import StringIO  # Add this import at the top of your file
import pickle
import pandas as pd
from src.interpretability.ice import compute_ice, plot_ice
from src.interpretability.pdp import compute_pdp, plot_pdp
from src.interpretability.surrogate_models import train_surrogate_model, evaluate_surrogate_model, plot_surrogate_performance
from src.interpretability.utilities import preprocess_data, compute_feature_importance


# Add the parent directory of repromodel_core to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statistics.normality_tests import shapiro_wilk_test, kolmogorov_smirnov_test, anderson_darling_test, qq_plot, histogram
from src.statistics.homogeneity_tests import levene_test, bartlett_test, box_plot
from src.statistics.sphericity_tests import mauchly_test, greenhouse_geisser_correction
from src.statistics.linearity_tests import scatter_plot, correlation_coefficient, durbin_watson_test
from src.statistics.parametric_tests import paired_t_test, two_sample_t_test, one_way_anova, repeated_measures_anova, linear_regression, f_test, z_test
from src.statistics.nonparametric_tests import wilcoxon_signed_rank_test, mann_whitney_u_test, kruskal_wallis_h_test, friedman_test, spearman_rank_correlation, chi_square_test, sign_test, kolmogorov_smirnov_test

import json
import ollama
import psutil
import logging
import subprocess
import requests
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from src.utils import print_to_file
import matplotlib.pyplot as plt
import base64
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
CORS(app)  # This will enable CORS for all routes

# Ensure the base directory exist.
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# Ensure the type directories exist.
for type_dir in TYPE_DIRS.values():
    path = os.path.join(BASE_DIR, type_dir)
    if not os.path.exists(path):
        os.makedirs(path)



######################################################################
# API ENDPOINTS - HEADER
######################################################################
# FUNCTION: Helper function to if process is running.
def is_process_running(process_name):
    # app.logger.info("process_name: %s", process_name)
   
    
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            # Join the command line arguments into a single string
            cmdline = ' '.join(proc.info.get('cmdline', []))
            if process_name in cmdline:
                return True, proc.info

        except Exception as e:
            logging.info("error: %s", e)

    return False, None


# Temp to get processes. Leave here for phase 3 when more sophisticated training controls are implemented
@app.route('/processes', methods=['GET'])
def list_processes():
    # Retrieve a list of all running processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'status', 'cmdline']):
        try:
            process_info = proc.info
            # Add command line information
            process_info['cmdline'] = ' '.join(proc.cmdline())
            processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return jsonify(processes)

# GET /ping
# Description: Returns whether backend is up and running.
@app.route('/ping', methods=['GET'])
def ping():

    # Check if training is in progress.
    trainingInProgress, process_info = is_process_running("trainer.py")
    #app.logger.info("trainingInProgress: %s", trainingInProgress)
   

    # Check if crossval testing is in progress.
    cvTestingInProgress, process_info = is_process_running("tester_crossval.py")
    # app.logger.info("testingInProgress: %s", testingInProgress)

    # Check if final testing is in progress.
    finalTestingInProgress, process_info = is_process_running("tester_final.py")
    # app.logger.info("testingInProgress: %s", testingInProgress)
    
    # Return HTTP 200 OK status code.
    return jsonify({ "message": "pong", 
                    "trainingInProgress": trainingInProgress, 
                    "cvTestingInProgress": cvTestingInProgress,
                    "finalTestingInProgress": finalTestingInProgress}), 200


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
        return jsonify({'message': 'File uploaded successfully.', 'path': save_path}), 200
    
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

        # Parse the JSON string back into a dictionary
        json_parsed = json.loads(json_data)

        with open("repromodel_core/last_experiment_config.json", 'w') as json_file:
            json.dump(json_parsed, json_file, indent=4)

        app.logger.info("Received JSON data for processing.")
        
        # Run the script trainer.py and capture the output.
        command = ['coverage', 'run', 'repromodel_core/trainer.py', json_data]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check the subprocess result.
        if result.returncode == 0:
            app.logger.info("Script executed successfully with output: %s", result.stdout)
            #return jsonify({'output': result.stdout, 'error': None})
        
        else: 
            error_detail = f"Training has been stopped: {result.stderr}"
            app.logger.error(error_detail)
            
            # Return HTTP 400 Bad Request status code.
            return jsonify({'output': result.stdout, 'error': error_detail}), 400
        
        # Saving coverage file
        os.makedirs('repromodel_core/extracted_code/', exist_ok=True)

        command = ['coverage', 'json', '-o', 'repromodel_core/extracted_code/coverage.json']
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check the subprocess result.
        if result.returncode == 0:
            app.logger.info("Coverage filenames successfully saved in the output folder with output: %s", result.stdout)
            return jsonify({'output': result.stdout, 'error': None})
        
        else: 
            error_detail = f"Coverage filename saving failed with error: {result.stderr}"
            app.logger.error(error_detail)
            
            # Return HTTP 400 Bad Request status code.
            return jsonify({'output': result.stdout, 'error': error_detail}), 400

    except Exception as e:
        
        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': error_message}), 500


# POST /kill-training-process
# Description: Kill the training process started from frontend.
@app.route('/kill-training-process', methods=['POST'])
def kill_training_process():

    try:        

        # Kill the training process.
        subprocess.run(['pkill', '-f', 'trainer.py'])
        app.logger.info("Process with name 'trainer' killed successfully.")

        return jsonify({'message': "Process with name 'trainer' killed successfully."})

    except Exception as e:

        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': error_message}), 500


# POST /copy-covered-files
# Description:
@app.route('/copy-covered-files', methods=['POST'])
def copy_files_endpoint():
    try:
        coverage_json_path = "repromodel_core/extracted_code/coverage.json"
        root_folder = "repromodel_core/extracted_code"
        additional_files = ["repromodel_core/tester_crossval.py", 
                            "repromodel_core/tester_final.py"
                            "repromodel_core/last_experiment_config.json",
                            "repromodel_core/last_crossVal_test_config.json",
                            "repromodel_core/last_final_test_config.json",
                            "repromodel_core/requirements.txt"]
        try:
            copy_covered_files(coverage_json_path, root_folder, additional_files)
            return jsonify({"status": "success", "message": "Files copied successfully"}), 200
        except Exception as e:
            return  jsonify({"status": "error", "message": "Copying files failed with error:" + str(e)}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# POST /create-repo
# Description: Creates a repository on user's GitHub profile      
@app.route('/create-repo', methods=['POST'])
def create_repo():
    try:
        github_token = request.form.get('github_token')
        repo_name = request.form.get('repo_name')
        description = request.form.get('description')
        privacy = request.form.get('privacy')

        local_directory = 'repromodel_core/extracted_code/repromodel_core'  # Ensure this directory is correct

        app.logger.info(f"Creating GitHub repository '{repo_name}'")

        # Create the repository on GitHub
        headers = {'Authorization': f'token {github_token}'}
        json_data = {
            'name': repo_name,
            'description': description,
            'private': True if privacy=="private" else False
        }
        response = requests.post('https://api.github.com/user/repos', headers=headers, json=json_data)
        
        if response.status_code == 201:
            message = f"Successfully created a {privacy} repository '{repo_name}' on GitHub"
            app.logger.info(message)
        else:
            app.logger.error(f"Error creating repository: {response.json()}")
            return jsonify({"status": "error", "message": f"Failed to create a GitHub repo. Make sure you entered the right API key and that the repository doesn't already exist. {response.json()}"}), 500

        # Initialize local repository and push to GitHub
        os.chdir(local_directory)
        run_command(['git', 'init'])
        run_command(['git', 'remote', 'add', 'origin', f'https://github.com/{response.json()["owner"]["login"]}/{repo_name}.git'])
        run_command(['git', 'add', '.'])
        run_command(['git', 'commit', '-m', 'Initial commit'])
        run_command(['git', 'branch', '-M', 'main'])  # Rename the default branch to 'main'
        run_command(['git', 'push', '-u', 'origin', 'main'])  # Push to the 'main' branch

        app.logger.info(f"Code from '{local_directory}' pushed to GitHub repository '{repo_name}'")
        return jsonify({"status": "success", "message": f'{message}. Code available at https://github.com/{response.json()["owner"]["login"]}/{repo_name}'}), 200

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def run_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        app.logger.info(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}")
        app.logger.error(e.stderr.decode('utf-8'))
        raise

######################################################################
# API ENDPOINTS - Model Testing
######################################################################

# POST /submit-config-start-crossValtesting
# Description: Start the testing process from frontend. CROSS VALIDATION!!!  
@app.route('/submit-config-start-crossValtesting', methods=['POST'])
def submit_config_start_crossValtesting_():
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

        # Parse the JSON string back into a dictionary
        json_parsed = json.loads(json_data)

        with open("repromodel_core/last_crossVal_test_config.json", 'w') as json_file:
            json.dump(json_parsed, json_file, indent=4)

        app.logger.info("Received JSON data for processing.")
        
        # Run the script tester_crossval.py and capture the output.
        command = ['python', 'repromodel_core/tester_crossval.py', json_data]
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


# POST /submit-config-start-finalValtesting
# Description: Start the testing process from frontend. FINAL TESTING!!!  
@app.route('/submit-config-start-finaltesting', methods=['POST'])
def submit_config_start_finaltesting_():
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

        # Parse the JSON string back into a dictionary
        json_parsed = json.loads(json_data)

        with open("repromodel_core/last_final_test_config.json", 'w') as json_file:
            json.dump(json_parsed, json_file, indent=4)

        app.logger.info("Received JSON data for processing.")
        
        # Run the script tester_final.py and capture the output.
        command = ['python', 'repromodel_core/tester_final.py', json_data]
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


# POST /kill-testing-process
# Description: Kill the testing process started from frontend.
@app.route('/kill-testing-process', methods=['POST'])
def kill_testing_process():

    try:        

        # Kill the testing process.
        subprocess.run(['pkill', '-f', 'tester_final.py'])
        subprocess.run(['pkill', '-f', 'tester_crossval.py'])
        app.logger.info("Process with name 'tester' killed successfully.")

        return jsonify({'message': "Process with name 'tester' killed successfully."})

    except Exception as e:

        error_message = f"An internal error occurred: {str(e)}"
        app.logger.exception(error_message)

        # Return HTTP 500 Internal Server Error status code.
        return jsonify({'error': error_message}), 500



######################################################################
# API ENDPOINTS - Progress Viewer
######################################################################


# FUNCTION: Helper function to start TensorBoard.
def start_tensorboard(logdir="logs", port=6006):
    # Check if there is a running TensorBoard instance.
    tensorboard_in_progress, process_info = is_process_running("tensorboard")
    if tensorboard_in_progress:
        return f"TensorBoard already running at http://localhost:{port} with logdir {logdir}"
    
    # Start a new TensorBoard instance.
    try:
        command = ['tensorboard', '--logdir', logdir, '--port', str(port), '--bind_all']
        tensorboard_proc = subprocess.Popen(command)
        print(tensorboard_proc)
        return f"TensorBoard started at http://localhost:{port} with logdir {logdir}"
    except Exception as e:
        return f"Failed to start TensorBoard: {str(e)}"

# GET /start-tensorboard
# Description: Start TensorBoard.
@app.route('/start-tensorboard', methods=['GET'])
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
# API ENDPOINTS - Statistical Tests
######################################################################


@app.route('/statistical_test', methods=['POST'])
def perform_statistical_test():
    data = request.json
    csv_data = data['csvData']
    test = data['test']
    params = data.get('params', {})

    try:
        df = pd.read_csv(StringIO(csv_data))
        columns = [df[col].values for col in df.columns]
        model_names = df.columns.tolist()

    except Exception as e:
        print_to_file(f"Error reading CSV data: {str(e)}")
        return jsonify({'error': f"Error reading CSV data: {str(e)}"}), 500  # More specific error message

    result = None
    plot = None
    try:
        if test in ['Shapiro-Wilk Test', 'Kolmogorov-Smirnov Test', 'Anderson-Darling Test', 'Q-Q Plot', 'Histogram']:
            result = []
            plot = []
            for i, col in enumerate(columns):
                if test == 'Shapiro-Wilk Test':
                    result.append(shapiro_wilk_test(col, params.get('alpha', 0.05)))
                elif test == 'Kolmogorov-Smirnov Test':
                    result.append(kolmogorov_smirnov_test(col, params.get('alpha', 0.05)))
                elif test == 'Anderson-Darling Test':
                    result.append(anderson_darling_test(col, params.get('alpha', 0.05)))
                elif test == 'Q-Q Plot':
                    plot.append(qq_plot(col))
                elif test == 'Histogram':
                    plot.append(histogram(col))
                
                if result:
                    result[-1]['model_name'] = model_names[i]
        elif test == 'Durbin-Watson Test':
            result = [durbin_watson_test(col) for col in columns]
            for i, res in enumerate(result):
                res['model_name'] = model_names[i]
        elif test == "Levene's Test":
            result = levene_test(*columns)
        elif test == "Bartlett's Test":
            result = bartlett_test(*columns)
        elif test == 'Box Plot':
            plot = box_plot(*columns, labels=model_names)
        elif test == "Mauchly's Test of Sphericity":
            result = mauchly_test(columns)
        elif test == 'Greenhouse-Geisser Correction':
            result = greenhouse_geisser_correction(columns)
        elif test == 'Scatter Plot':
            if len(columns) != 2:
                return jsonify({'error': 'Scatter Plot requires exactly two columns'}), 400
            plot = scatter_plot(columns[0], columns[1])
        elif test == 'Correlation Coefficient':
            if len(columns) != 2:
                return jsonify({'error': 'Correlation Coefficient requires exactly two columns'}), 400
            result = correlation_coefficient(columns[0], columns[1])
        elif test == 'Paired t-test':
            if len(columns) != 2:
                return jsonify({'error': 'Paired t-test requires exactly two columns'}), 400
            result = paired_t_test(columns[0], columns[1], params.get('alpha', 0.05))
        elif test == 'Two-sample t-test':
            if len(columns) != 2:
                return jsonify({'error': 'Two-sample t-test requires exactly two columns'}), 400
            result = two_sample_t_test(columns[0], columns[1], params.get('alpha', 0.05), params.get('equal_var', True))
        elif test == 'Analysis of Variance (ANOVA)':
            result = one_way_anova(*columns)
        elif test == 'Repeated Measures ANOVA':
            result = repeated_measures_anova(*columns)
        elif test == 'Linear Regression Analysis':
            if len(columns) != 2:
                return jsonify({'error': 'Linear Regression Analysis requires exactly two columns'}), 400
            result = linear_regression(columns[0], columns[1])
        elif test == 'F-test for comparing variances':
            if len(columns) != 2:
                return jsonify({'error': 'F-test requires exactly two columns'}), 400
            result = f_test(columns[0], columns[1])
        elif test == 'Z-test':
            if len(columns) != 1:
                return jsonify({'error': 'Z-test requires exactly one column'}), 400
            result = z_test(columns[0], params.get('population_mean'), params.get('population_std'))
        elif test == 'Wilcoxon Signed-Rank Test':
            if len(columns) != 2:
                return jsonify({'error': 'Wilcoxon Signed-Rank Test requires exactly two columns'}), 400
            result = wilcoxon_signed_rank_test(columns[0], columns[1], params.get('alpha', 0.05))
        elif test == 'Mann-Whitney U Test':
            if len(columns) != 2:
                return jsonify({'error': 'Mann-Whitney U Test requires exactly two columns'}), 400
            result = mann_whitney_u_test(columns[0], columns[1], params.get('alpha', 0.05))
        elif test == 'Kruskal-Wallis H Test':
            result = kruskal_wallis_h_test(*columns)
        elif test == 'Friedman Test':
            result = friedman_test(*columns)
        elif test == "Spearman's Rank Correlation":
            if len(columns) != 2:
                return jsonify({'error': "Spearman's Rank Correlation requires exactly two columns"}), 400
            result = spearman_rank_correlation(columns[0], columns[1])
        elif test == 'Chi-Square Test':
            if len(columns) != 2:
                return jsonify({'error': 'Chi-Square Test requires exactly two columns'}), 400
            result = chi_square_test(columns[0], columns[1])
        elif test == 'Sign Test':
            if len(columns) != 2:
                return jsonify({'error': 'Sign Test requires exactly two columns'}), 400
            result = sign_test(columns[0], columns[1])
        elif test == 'Kolmogorov-Smirnov Test':
            if len(columns) != 2:
                return jsonify({'error': 'Kolmogorov-Smirnov Test requires exactly two columns'}), 400
            result = kolmogorov_smirnov_test(columns[0], columns[1])
        else:
            return jsonify({'error': 'Unsupported test'}), 400
    except Exception as e:
        print_to_file(f"Error performing statistical test: {str(e)}")
        return jsonify({'error': f"Error performing statistical test: {str(e)}"}), 500  # More specific error message


    if isinstance(result, list):
        for i, res in enumerate(result):
            res['model_name'] = model_names[i]
    elif result is None:
        result = {}
    else:
        result['model_names'] = model_names

    # Convert numpy boolean to Python boolean
    if isinstance(result, np.bool_):
        result = bool(result)

    if plot is not None:
        # Convert the plot to a base64-encoded string
        try:
            return jsonify({'result': result, 'plot': plot})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'result': convert_numpy_types(result)})

def convert_numpy_types(data):
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    return data

# interpretability

@app.route('/interpretability', methods=['POST'])
def interpretability():
    model_file = request.files['model']
    data_file = request.files['data']
    method = request.form['method']
    feature_names = request.form['feature_names'].split(',')
    target_name = request.form['target_name']

    model_path = 'uploaded_model.pkl'
    data_path = 'uploaded_data.csv'

    model_file.save(model_path)
    data_file.save(data_path)

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.npy'):
        data = np.load(data_path)
        data = pd.DataFrame(data, columns=feature_names)
    else:
        return jsonify({'error': 'Unsupported data file format.'}), 400

    X = data[feature_names]
    y = data[target_name]

    results = None
    plot_image = None
    feature_importance = None

    try:
        if method == 'Individual Conditional Expectation (ICE)':
            feature = request.form['feature']
            grid_resolution = int(request.form['grid_resolution'])
            percentiles = tuple(map(float, request.form['percentiles'].split(',')))
            kind = request.form['kind']
            subsample = float(request.form['subsample'])
            random_state = int(request.form['random_state']) if request.form['random_state'] else None

            ice_disp = compute_ice(model, X, feature, grid_resolution=grid_resolution, percentiles=percentiles,
                                   kind=kind, subsample=subsample, random_state=random_state)
            fig, _ = plot_ice(ice_disp, feature_name=feature)
            plot_image = fig_to_base64(fig)
        elif method == 'Partial Dependence Plot (PDP)':
            pdp_features = request.form['pdp_features'].split(',')
            pdp_kind = request.form['pdp_kind']
            pdp_grid_resolution = int(request.form['pdp_grid_resolution'])

            pdp_disp = compute_pdp(model, X, features=pdp_features, kind=pdp_kind, grid_resolution=pdp_grid_resolution)
            fig, _ = plot_pdp(pdp_disp, feature_names=pdp_features)
            plot_image = fig_to_base64(fig)
        elif method == 'Surrogate Models':
            surrogate_model_type = request.form['surrogate_model_type']
            max_depth = int(request.form['max_depth'])
            n_estimators = int(request.form['n_estimators'])
            plot_performance = request.form['plot_performance'] == 'true'
            sample_size = int(request.form['sample_size'])

            X_train, _, _ = preprocess_data(X, y)
            surrogate_model = train_surrogate_model(model, X_train, y, model_type=surrogate_model_type,
                                                    max_depth=max_depth, n_estimators=n_estimators)
            results = evaluate_surrogate_model(model, surrogate_model, X)
            if plot_performance:
                fig = plot_surrogate_performance(model, surrogate_model, X, sample_size=sample_size)
                plot_image = fig_to_base64(fig)
        elif method == 'Feature Importance':
            fi_feature_names = request.form['fi_feature_names'].split(',')
            X_train, _, _ = preprocess_data(X, y)
            feature_importance = compute_feature_importance(model, fi_feature_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(model_path)
        os.remove(data_path)

    return jsonify({
        'results': results,
        'plot_image': plot_image,
        'feature_importance': feature_importance
    })

def fig_to_base64(fig):
    from io import BytesIO
    import base64

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

######################################################################
# MAIN METHOD
######################################################################


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True, debug=True, use_reloader=True)