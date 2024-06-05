import "./status-checks.css"

import axios from "axios"
import React from "react"

import { useEffect, useState } from "react"

const StatusCheck = () => {
    
  const [isBackendActive, setIsBackendActive] = useState(false)

  useEffect(() => {
    
    const interval = setInterval(() => {
      
      axios.get("http://127.0.0.1:5005/ping")
        
        .then(response => {
          if (response.status === 200) {
            setIsBackendActive(true)
          } else {
            setIsBackendActive(false)
          }
        })      

        .catch(error => {
          setIsBackendActive(false)
        })

    }, 3000)

    return () => clearInterval(interval)

  }, [])
    
    

  return (
    <div className = "status-wrapper-position">

      <div className = "blinking"
        style = { {
          width: "10px",
          height: "10px",
          borderRadius: "50%",
          backgroundColor: isBackendActive ? "green" : "red",
          marginRight: "10px", 
          marginLeft: "8px"
        }}
      />

      <span>
        { isBackendActive ? "Backend Active" : "Backend Offline" }
      </span>
    
    </div>
  )
}

export default StatusCheck