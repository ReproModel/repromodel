import React from 'react'

const MobileWarning = () => {

  return (
    
    <div className = "container">
      
      <h1 className = "header">Oops!</h1>
      
      <p className = "message">ReproModel is currently only built for Desktop use.</p>
      <p className = "message">Looks like you're trying to access it on a mobile device. Trust us, it's way better on a big screen!</p>
      
      <p className = "footer">Please visit us from a desktop for the full experience.</p>
    
    </div>
  )
}

export default MobileWarning