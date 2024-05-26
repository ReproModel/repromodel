import axios from 'axios'
import React from 'react'

import { Box, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import { useEffect, useState } from 'react'

const handleChange = (event) => {
  const value = event.target.value
  setSelectedFile(value)
  onSelectFile(value)
}

const FileDropdown = ({ onSelectFile }) => {
  
  const [files, setFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState('')

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5005/api/files')
        setFiles(response.data);
      } catch (error) {
        console.error('Error fetching files: ', error)
      }
    }
    fetchFiles()
  }, [])

  return (
    <Box sx = { { width: 300, mt: 4 } }>
      
      <FormControl fullWidth>
        
        <InputLabel id = "file-select-label">Output File</InputLabel>
        
        <Select
          labelId = "file-select-label"
          value = { selectedFile }
          label = "Select a file"
          onChange = { handleChange }
        >
          { files.map((file, index) => (
            <MenuItem key = { index } value = { file }>{ file }</MenuItem>
          ))}
        </Select>

      </FormControl>

    </Box>
  )
}

export default FileDropdown