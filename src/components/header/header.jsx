import FirstTimeModal from "../first-time-modal/first-time-modal"
import "./header.css"

import { Box, Typography } from "@mui/material"

function Header() {
  
  return (
    <> 
      
      <Box className = "header">
      <FirstTimeModal/>
        <Typography variant = "h2" className = "title">ReproModel</Typography>
        <Typography variant = "h6" className = "title">Reproducable and Comparable AI</Typography>  
      
      </Box>
    </>
  )
}

export default Header