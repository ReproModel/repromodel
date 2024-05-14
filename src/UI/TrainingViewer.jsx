import React from "react";
import { useState, useEffect } from "react";
import axios from "axios";
import FileDropdown from "./ViewerComponents/FileDropdown";
import ProgressFileReader from "./ViewerComponents/ProgressFileReader";



const RunPythonScript = () => {
  const runScript = async () => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/run-python-script"
      );
      console.log("Script Output:", response.data);
    } catch (error) {
      console.error("Error running script:", error);
    }
  };

  return (
    <div>
      <button onClick={runScript}>Run Python Script</button>
    </div>
  );
};

const TrainingViewer = ({}) => {
  const [selectedFile, setSelectedFile] = useState("");
  return (
    <>
      <RunPythonScript />
      <FileDropdown onSelectFile={setSelectedFile} />
      {selectedFile && <ProgressFileReader fileName={selectedFile} />}
    </>
  );
};

export default TrainingViewer;
