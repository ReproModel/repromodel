import React from "react"

import { capitalizeFirstLetter } from "../../utils/string-helpers"
import { Paper, Typography } from "@mui/material"

function ActiveBlock({ id, name, status }) {
  return (
    <Paper
      key = { id }
      variant = "outlined"
      sx = { { backgroundColor: "#95e9fb", p: 0.5, boxShadow: "0 0 10px rgba(0, 255, 0, 0.5)" } }
    >
      <Typography align = "center" variant = "h6" gutterBottom>{ capitalizeFirstLetter(name) }</Typography>
    </Paper>
  )
}

function CompletedBlock({ id, name, status }) {
  return (
    <Paper
      key = { id }
      variant = "outlined"
      sx = { { backgroundColor: "#dff9fe", p: 0.5 } }
    >
      <Typography align = "center" variant = "h6" gutterBottom>{ capitalizeFirstLetter(name) }</Typography>
    </Paper>
  )
}

function PassiveBlock({ id, name, status }) {
  return (
    <Paper
      key = { id }
      variant = "outlined"
      sx = { { backgroundColor: "#e6f0f1", p: 0.5 } }
    >
      <Typography align = "center" variant = "h6" gutterBottom>{ capitalizeFirstLetter(name) }</Typography>
    </Paper>
  )
}

function FlexibleBlock({ status, name }) {
  
  const renderSwitch = () => {
    
    switch (status) {
      
      case "active":
        return <ActiveBlock id = { name } name = { name } status = { status } />
      
      case "completed":
        return <CompletedBlock id = { name } name = { name } status = { status } />
      
      case "passive":
        return <PassiveBlock id = { name } name = { name } status = { status } />
      
      default:
        return <div>Error occurred - unsupported type.</div>

    }

  }

  return <>{ renderSwitch() }</>
}

export default FlexibleBlock