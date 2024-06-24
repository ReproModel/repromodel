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
      {/* Header - TensorBoard */}
      <div className = "header-tensorboard">
          <Typography variant = "h7" style = { { fontWeight: "600" } }>TensorBoard</Typography>
      </div>

      {/* Subheader - TensorBoard */}
      <div className = "subheader-tensorboard">
          <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>A suite of visualization tools to understand, debug, and optimize ML experiments.</span>
      </div>

      <button className = "tensorboard-button" onClick = { startTensorBoard }>
        <Typography>Start TensorBoard</Typography>
      </button>
      
      {/* Header - View Output */}
      <div className = "header-output">
          <Typography variant = "h7" style = { { fontWeight: "600" } }>View Output</Typography>
      </div>

      {/* Subheader - View Output */}
      <div className = "subheader-output">
          <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>View output logs and training progress.</span>
      </div>
      
      {/* File Input - View Output */}
      <div className = "file-input-output">
        <FileDropdown onSelectFile = { setSelectedFile }/>
      </div>

      <div className = "file-reader">
        { selectedFile && <FileReader fileName = { selectedFile }/> }
      </div>

    </>
  )
}

export default ProgressViewer