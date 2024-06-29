import "./flexible-block.css"

import React from "react"

import { capitalizeAndRemoveUnderscore } from "../../../utils/string-helpers"
import { Paper, Typography } from "@mui/material"

function ActiveBlock({ id, name, status }) {
  return (
    <Paper
      key = { id }
      variant = "outlined"
      sx = { { backgroundColor: "#95e9fb", height: "33px", boxShadow: "0 0 10px rgba(0, 255, 0, 0.5)" } }
    >
      <Typography align = "center" style = { { fontWeight: "500", fontSize: "16px" } } gutterBottom>{ capitalizeAndRemoveUnderscore(name) }</Typography>
    </Paper>
  )
}

function CompletedBlock({ id, name, status }) {
  return (
    <Paper
      key = { id }
      variant = "outlined"
      sx = { { backgroundColor: "#dff9fe", height: "33px" } }
    >
      <Typography align = "center" style = { { fontWeight: "500", fontSize: "16px" } } gutterBottom>{ capitalizeAndRemoveUnderscore(name) }</Typography>
    </Paper>
  )
}

function PassiveBlock({ id, name, status }) {

  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])

  return (
    <Paper
      key = { id }
      variant = "outlined"
      sx = { { backgroundColor: (isDarkTheme ? "hsl(0, 0%, 20%)" : "#e6f0f1"), height: "33px" } }
    >
      <Typography align = "center" className = "passive-block-text" gutterBottom>{ capitalizeAndRemoveUnderscore(name) }</Typography>
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