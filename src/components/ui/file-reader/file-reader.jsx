import "./file-reader.css"

import React from "react"

import { useState, useEffect } from "react"

const FileReader = ({ fileName }) => {

  const [fileContent, setFileContent] = useState("")

  useEffect(() => {
    const fetchFileContent = async () => {
      try {
        const response = await fetch(`repromodel_core/logs/${fileName}`)
        const text = await response.text()
        setFileContent(text)
      } catch (error) {
        console.error("Error fetching the file:", error)
      }
    }
    fetchFileContent()

    // Update every 1000 ms.
    const interval = setInterval(fetchFileContent, 1000)

    // Clean up the interval on component unmount.
    return () => clearInterval(interval)

  }, [fileName])

  return (
    <>
      <h4>Progress</h4>
      
      <div className = "file-content"> 
        <pre>{ fileContent }</pre>
      </div>
    </>
  )
}

export default FileReader