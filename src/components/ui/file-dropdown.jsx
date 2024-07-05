import axios from "axios"
import React from "react"

import { Box, FormControl, InputLabel, MenuItem, Select } from "@mui/material"
import { useEffect, useState } from "react"

const FileDropdown = ({ onSelectFile }) => {

  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])
  
  const [files, setFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState("command_output.txt")

  // Load command_output.txt on page load.
  useEffect(() => {
    onSelectFile("command_output.txt")
  }, [])

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
    <Box sx = { { width: 300, mt: 4, borderRadius: 2, backgroundColor: (isDarkTheme ? "hsl(0, 0%, 20%)" : "white") } }>
      
      <FormControl fullWidth>
        
        <InputLabel id = "file-select-label" style = { { color: (isDarkTheme ? "white" : "black") } }>Select file...</InputLabel>
        
        <Select
          labelId = "file-select-label"
          label = "Select a file"
          value = { selectedFile }
          onChange = { handleChange }
          style = { { color: (isDarkTheme ? "white" : "black") } }
        >
          { files.map((file, index) => (
            <MenuItem key = { index } value = { file }>
              <span>{ file }</span>
            </MenuItem>
          ))}
        </Select>

      </FormControl>

    </Box>
  )
}

export default FileDropdown