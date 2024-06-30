import React from "react"

import { Field } from "formik"

export default function DefaultTextField({ id, label, name, type }) {

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])

  return (
    <Field
      className = "inputField"
      id = { id }
      name = { name }
      label = { label }
      placeholder = { `Please enter ${type}` }
      style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black" } }
    />
  )
}