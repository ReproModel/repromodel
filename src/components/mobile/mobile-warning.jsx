import "./mobile-warning.css"

import React from 'react'

const MobileWarning = () => {

  return (
    
    <div className = "mobile-warning-container">

      <div className = "mobile-warning-image-container">
      
        <h1 className = "mobile-warning-header">Oops!</h1>
        
        <p className = "mobile-warning-message">ReproModel is currently only built for Desktop use.</p>
        <p className = "mobile-warning-message">Looks like you're trying to access it on a mobile device. Trust us, it's way better on a big screen!</p>
        
        <p className = "mobile-warning-footer">Please visit us from a desktop for the full experience.</p>
      
      </div>
    
    </div>
  )
}

export default MobileWarning