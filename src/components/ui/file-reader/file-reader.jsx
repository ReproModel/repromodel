import "./file-reader.css"

import React, { useState, useEffect, useRef } from "react"

const FileReader = ({ fileName }) => {

  const [fileContent, setFileContent] = useState("")
  const fileContentRef = useRef(null)

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

  useEffect(() => {
    if (fileContentRef.current) {
      fileContentRef.current.scrollTop = fileContentRef.current.scrollHeight
    }
  }, [fileContent])

  return (
    <>
      <h4 className = "file-reader-header">Progress</h4>
      
      <div className = "file-content" ref={fileContentRef}> 
        <pre>{ fileContent }</pre>
      </div>
    </>
  )
}

export default FileReader