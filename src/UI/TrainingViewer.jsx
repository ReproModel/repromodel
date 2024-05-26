import axios from "axios"
import FileDropdown from "./ViewerComponents/FileDropdown"
import ProgressFileReader from "./ViewerComponents/ProgressFileReader"
import React from "react"

import { useState } from "react"

const TrainingViewer = ({}) => {

  const [selectedFile, setSelectedFile] = useState("")
  
  const startTensorBoard = () => {
    
    // URL of Flask endpoint.
    axios.get('http://localhost:5005/start-tensorboard')
      .then(response => {
        
        // Display response data message.
        alert(response.data.message)
        
        // Open TensorBoard in a new tab.
        window.open('http://localhost:6006', '_blank')
      })
      .catch(error => {
        
        // Log TensorBoard error.
        console.error('Error starting TensorBoard: ', error)

        // Display TensorBoard error occured.
        alert('Failed to start TensorBoard. Check console for more details.')
      })
  }

  return (
    <>
      <button onClick = { startTensorBoard }>Start TensorBoard</button>
      
      <FileDropdown onSelectFile = { setSelectedFile } />
      
      { selectedFile && <ProgressFileReader fileName = { selectedFile } /> }
    </>
  )
}

export default TrainingViewer
