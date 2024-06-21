import FirstTimeModal from "../first-time-modal/first-time-modal"
import StatusCheck from "../status-checks/status-checks"
import "./header.css"

import { Box, Typography } from "@mui/material"

function Header() {
  
  return (
    <> 
      
      <Box className = "header">
      <FirstTimeModal/>
      <StatusCheck/>
      
        <Typography variant = "h3" className = "title">ReproModel</Typography>
        <Typography variant = "h6" className = "title">Reproducible and Comparable AI</Typography>  
      
      </Box>
    </>
  )
}

export default Header