import React from "react"

import { useEffect, useState } from "react"

const ProgressFileReader = ({ fileName }) => {
  
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
    
    // Fetch file content.
    fetchFileContent()
    
    // Update file content every 1000 ms.
    const interval = setInterval(fetchFileContent, 1000)

    // Clean up the interval on component unmount.
    return () => clearInterval(interval)

  }, [fileName])

  return (
    <>
      <h4>Progress</h4>
      
      <div style = { { width: "95%", marginTop: "16px", height: "55%", border: "2px solid #000", padding: "10px", overflow: "auto", borderRadius: "10px" } }> 
        <pre>{ fileContent }</pre>
      </div>
    </>
  )
}

export default ProgressFileReader