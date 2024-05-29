import "./progress-viewer.css"

import axios from "axios"
import FileDropdown from "../ui/file-dropdown"
import FileReader from "../ui/file-reader/file-reader"
import React from "react"

import { Typography } from "@mui/material"
import { useState } from "react"

const ProgressViewer = ({}) => {

  const [selectedFile, setSelectedFile] = useState("")
  
  const startTensorBoard = () => {
    
    // Send GET request to /start-tensorboard Flask endpoint.
    axios.get("http://localhost:5005/start-tensorboard")
      .then(response => {
        
        // Display response data message.
        alert(response.data.message)
        
        // Open TensorBoard in a new tab.
        window.open("http://localhost:6006", "_blank")
      })
      .catch(error => {
        
        // Log TensorBoard error in console.
        console.error("Error starting TensorBoard: ", error)

        // Display TensorBoard error occured.
        alert("Failed to start TensorBoard. Check console for more details.")
      })
  }

  return (
    <>
      <button className = "tensorboard-button" onClick = { startTensorBoard }>
        <Typography>Start TensorBoard</Typography>
      </button>
      
      <FileDropdown onSelectFile = { setSelectedFile }/>
    
      { selectedFile && <FileReader fileName = { selectedFile }/> }
    </>
  )
}

export default ProgressViewer