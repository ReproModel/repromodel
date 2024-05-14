// src/components/FileDropdown.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { FormControl, InputLabel, Select, MenuItem, Box } from '@mui/material';

const FileDropdown = ({ onSelectFile }) => {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/api/files');
        setFiles(response.data);
      } catch (error) {
        console.error('Error fetching files:', error);
      }
    };

    fetchFiles();
  }, []);

  const handleChange = (event) => {
    const value = event.target.value;
    setSelectedFile(value);
    onSelectFile(value);
  };

  return (
    <Box sx={{ width: 300, mt: 4 }}>
      <FormControl fullWidth>
        <InputLabel id="file-select-label">Training Run</InputLabel>
        <Select
          labelId="file-select-label"
          value={selectedFile}
          label="Select a file"
          onChange={handleChange}
        >
          {files.map((file, index) => (
            <MenuItem key={index} value={file}>
              {file}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );
};

export default FileDropdown;
