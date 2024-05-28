import axios from "axios"
import FileDropdown from "./file-dropdown"
import ProgressFileReader from "./progress-file-reader"
import React from "react"

import { useState } from "react"
import { Typography } from "@mui/material"

const ProgressViewer = ({}) => {

  const [selectedFile, setSelectedFile] = useState("")
  
  const startTensorBoard = () => {
    
    // Send GET request to /start-tensorboard Flask endpoint.
    axios.get('http://localhost:5005/start-tensorboard')
      .then(response => {
        
        // Display response data message.
        alert(response.data.message)
        
        // Open TensorBoard in a new tab.
        window.open('http://localhost:6006', '_blank')
      })
      .catch(error => {
        
        // Log TensorBoard error in console.
        console.error('Error starting TensorBoard: ', error)

        // Display TensorBoard error occured.
        alert('Failed to start TensorBoard. Check console for more details.')
      })
  }

  return (
    <>
      <button className = "tensorboard-button" onClick = { startTensorBoard }>
        <Typography>Start TensorBoard</Typography>
      </button>
      
      <FileDropdown onSelectFile = { setSelectedFile } />
    
      { selectedFile && <ProgressFileReader fileName = { selectedFile } /> }
    </>
  )
}

export default ProgressViewer