import axios from "axios"
import React from "react"

import { Box, FormControl, InputLabel, MenuItem, Select } from "@mui/material"
import { useEffect, useState } from "react"

const FileDropdown = ({ onSelectFile }) => {
  
  const [files, setFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState("")

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5005/api/files")
        setFiles(response.data)
      } catch (error) {
        console.error("Error fetching files:", error)
        setFiles(["To see files here...", "...start the backend", "...and run a training"])
      }
    }
    fetchFiles()
  }, [])

  const handleChange = (event) => {
    const value = event.target.value
    setSelectedFile(value)
    onSelectFile(value)
  }

  return (
    <Box sx = { { width: 300, mt: 4 } }>
      
      <strong>Output</strong>
      
      <FormControl fullWidth>
        
        <InputLabel id = "file-select-label">Select file...</InputLabel>
        
        <Select
          labelId = "file-select-label"
          label="Select a file"
          value = { selectedFile }
          onChange = { handleChange }
        >
          { files.map((file, index) => (
            <MenuItem key = { index } value = { file }>
              { file }
            </MenuItem>
          ))}
        </Select>

      </FormControl>

    </Box>
  )
}

export default FileDropdown