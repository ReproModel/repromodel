import "./first-time-modal.css"

import axios from "axios"
import Button from "@mui/material/Button"
import Box from "@mui/material/Box"
import CloseIcon from "@mui/icons-material/Close"
import IconButton from "@mui/material/IconButton"
import Modal from "@mui/material/Modal"
import React from "react"
import experimentConfig from "../../../repromodel_core/experiment_config.json"
import Typography from "@mui/material/Typography"

import { handleDownload } from "../../utils/download-helpers"
import { useState } from "react"

const FirstTimeModal = () => {
    
  const [open, setOpen] = useState(false)
  const [confirmationMessage, setConfirmationMessage] = useState("")

  const handleOpen = () => setOpen(true)
  
  const handleClose = () => {
    setOpen(false)

    // Reset the confirmation message when the modal is closed.
    setConfirmationMessage("")
  }
  
  const handleGenerate = async () => {
    try {
      const response = await axios.get("http://localhost:5005/generate-dummy-data")
      console.log("Data generated:", response.data)
      setConfirmationMessage("Dummy data generated successfully!")
    
    } catch (error) {
      console.error("Error generating dummy data: ", error)
      setConfirmationMessage("Failed to generate dummy data.")
    }
  }

  return (
    <div className = "modal-container">
      
      <Button variant = "contained" onClick = { handleOpen } style = { { backgroundColor: "#1b1212", opacity: "90%", borderRadius: "24px" } }>
        First Time Here?
      </Button>
      
      <Modal open = { open } onClose = { handleClose }>
        
        <Box className = "modal">
          
          {/* Close Button */}
          <IconButton onClick = { handleClose } style = { { position: "absolute", right: 8, top: 8 } }>
            <CloseIcon />
          </IconButton>

          {/* Title */}
          <Typography id = "modal-title" variant = "h5" component = "h2" className = "modal-title">
            Quick Start
          </Typography>
          
          {/* Subtitle */}
          <Typography id = "modal-description" className = "modal-description">
            To get started with ReproModel, try loading some default data.
          </Typography>
          
          {/* Step 1 - Generate Dummy Data */}
          <Typography className = "modal-step-one">
            <strong>1. Generate dummy data.</strong>
            <Typography style = { { fontStyle: "italic", opacity: "40%", fontSize: "14px", paddingTop: "6px" } }>repromodel_core/data/dummyData/input</Typography>
          </Typography>
          
          <Button variant = "contained" className = "modal-button" onClick = { handleGenerate }>
            Generate
          </Button>
          
          {/* Step 2 - Download Configuration */}
          <Typography className = "modal-step-two">
            <strong>2. Download configuration file.</strong>
            <Typography style = { { fontStyle: "italic", opacity: "40%", fontSize: "14px", paddingTop: "6px" } }>DemoConfig.json</Typography>
          </Typography>
          
          <Button variant = "contained" className = "modal-button" onClick = { () => handleDownload(experimentConfig, 'DemoConfig', 'json') }>
            Download
          </Button>
                  
          {/* Confirmation Message */}
          { confirmationMessage && (
            <Typography className = "modal-confirmation" style = { { color: 'green' } }>
              { confirmationMessage }
            </Typography>
          )}

          {/* Step 3 - Upload */}
          <Typography className = "modal-step-two">
            <strong>3. Upload the downloaded file in Experiment Builder.</strong>
          </Typography>

          <img src = "file_input.jpg" style = { { width: "75%" } } />

          {/* Parting Message */}
          <Typography className = "modal-message">Happy developing with ReproModel 😎</Typography>

        </Box>

      </Modal>

    </div>
  )
}

export default FirstTimeModal