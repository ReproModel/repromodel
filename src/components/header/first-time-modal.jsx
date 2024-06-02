import React, { useState } from 'react';
import Button from '@mui/material/Button';
import Modal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import axios from 'axios';
import './first-time-modal.css';
import trainConfig from "../../../repromodel_core/train_config.json"

const FirstTimeModal = () => {
    const [open, setOpen] = useState(false);
    const [confirmationMessage, setConfirmationMessage] = useState('');
  
    const handleOpen = () => setOpen(true);
    const handleClose = () => {
      setOpen(false);
      setConfirmationMessage(''); // Reset the confirmation message when the modal is closed
    };
  
    const handleGenerate = async () => {
      try {
        const response = await axios.get('http://localhost:5005/generate-dummy-data');
        console.log('Data generated:', response.data);
        setConfirmationMessage('Dummy data generated successfully!');
      } catch (error) {
        console.error('Error generating dummy data:', error);
        setConfirmationMessage('Failed to generate dummy data.');
      }
    };
  
    const handleDownload = () => {
      
      const json = JSON.stringify(trainConfig, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const href = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = href;
      link.download = 'DemoTrainingConfig.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };

  return (
    <div className="wrapper-position">
      <Button  variant="contained" onClick={handleOpen}>
        First Time Here?
      </Button>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box className="modal-style">
          <Typography id="modal-modal-title" variant="h6" component="h2" className="modal-content">
            Hey, nice to have you here
          </Typography>
          <Typography id="modal-modal-description" className="modal-content">
            For a quick start and to try out repromodel you can load some defaults.
          </Typography>
          <Typography className="modal-content">
            1. Generate Dummy Data
          </Typography>
          <Button variant="contained" className="modal-button" onClick={handleGenerate}>
            Generate
          </Button>
          <Typography className="modal-content">
            2. Download a training script
          </Typography>
          <Button variant="contained" className="modal-button" onClick={handleDownload}>
            Download
          </Button>
          {confirmationMessage && (
            <Typography className="modal-content" style={{ color: 'green' }}>
              {confirmationMessage}
            </Typography>
          )}
          <Typography className="modal-content">
            Happy Developing with Repromodel
          </Typography>
        </Box>
      </Modal>
    </div>
  );
};

export default FirstTimeModal;