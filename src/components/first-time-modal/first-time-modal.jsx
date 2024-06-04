import "./first-time-modal.css"

import axios from "axios"
import Button from "@mui/material/Button"
import Box from "@mui/material/Box"
import CloseIcon from "@mui/icons-material/Close"
import IconButton from "@mui/material/IconButton"
import Modal from "@mui/material/Modal"
import React from "react"
import trainConfig from "../../../repromodel_core/train_config.json"
import Typography from "@mui/material/Typography"

import { handleDownload } from "../../utils/download-helpers";
import { useState } from "react";

const FirstTimeModal = () => {
    
  const [open, setOpen] = useState(false)
  const [confirmationMessage, setConfirmationMessage] = useState("")

  const handleOpen = () => setOpen(true)
  
  const handleClose = () => {
    setOpen(false)

    // Reset the confirmation message when the modal is closed
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
      
      <Button variant = "contained" onClick = { handleOpen }>First Time Here?</Button>
      
      <Modal open = { open } onClose = { handleClose }>
        
        <Box className = "modal">
          
          <IconButton onClick = { handleClose } style = { { position: "absolute", right: 8, top: 8 } }>
            <CloseIcon />
          </IconButton>

          <Typography id = "modal-title" variant = "h6" component = "h2" className = "modal-title">Hey, nice to have you here!</Typography>
          
          <Typography id = "modal-description" className = "modal-description">For a quick start and to try out ReproModel you can load some defaults.</Typography>
          
          <Typography className = "modal-step-one">1. Generate dummy data.</Typography>          
          <Button variant = "contained" className = "modal-button" onClick = { handleGenerate }>Generate</Button>

          <Typography className = "modal-step-two">2. Download a training script.</Typography>
          <Button variant = "contained" className = "modal-button" onClick = { () => handleDownload(trainConfig, "DemoTrainingConfig", "json") }>Download</Button>
          
          { confirmationMessage && (
            <Typography className = "modal-confirmation" style = { { color: "green" } }>
              { confirmationMessage }
            </Typography>
          )}

          <Typography className = "modal-message">Happy developing with ReproModel.</Typography>
        
        </Box>
      
      </Modal>
    
    </div>
  )
}

export default FirstTimeModal