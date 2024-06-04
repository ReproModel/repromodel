import React, { useState, useEffect} from 'react';
import Button from '@mui/material/Button';
import Modal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import axios from 'axios';
import './status-checks.css';
import trainConfig from "../../../repromodel_core/train_config.json"


const StatusCheck = () => {
    
    const [isBackendActive, setIsBackendActive] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      axios.get("http://127.0.0.1:5005/ping")
        .then(response => {
          if (response.status === 200) {
            setIsBackendActive(true);
          } else {
            setIsBackendActive(false);
          }
        })
        .catch(error => {
          setIsBackendActive(false);
        });
    }, 3000);

    return () => clearInterval(interval);
  }, []);
    
    

  return (
    <div className="status-wrapper-position">

      <div className='blinking'
        style={{
          width: '10px',
          height: '10px',
          borderRadius: '50%',
          backgroundColor: isBackendActive ? 'green' : 'red',
          marginRight: '10px', 
          marginLeft: "8px",
        }}
      />
      <span>
        {isBackendActive ? 'Backend Active' : 'Backend Offline'}
      </span>
    
    </div>
  );
};

export default StatusCheck;