import React from "react";
import { useState, useEffect } from "react";
import axios from "axios";
import FileDropdown from "./ViewerComponents/FileDropdown";
import ProgressFileReader from "./ViewerComponents/ProgressFileReader";


const TrainingViewer = ({}) => {
  const startTensorBoard = () => {
    // URL of the Flask endpoint
    axios.get('http://localhost:5005/start-tensorboard')
      .then(response => {
        // Handle response here
        alert(response.data.message);
        // Opens TensorBoard in a new tab
        window.open('http://localhost:6006', '_blank');
      })
      .catch(error => {
        // Handle error here
        console.error('Error starting TensorBoard:', error);
        alert('Failed to start TensorBoard. Check console for more details.');
      });
  };
  const [selectedFile, setSelectedFile] = useState("");
  return (
    <>
      <button onClick={startTensorBoard}>Start TensorBoard</button>
      <FileDropdown onSelectFile={setSelectedFile} />
      {selectedFile && <ProgressFileReader fileName={selectedFile} />}
    </>
  );
};

export default TrainingViewer;
