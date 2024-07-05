import React from "react"

import { Field } from "formik"

function FloatField({ id, label, object, name }) {

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])
    
  // Extracted and parsed the range from the object, if it exists by using optional chaining.
  const rangeString = object?.range
  let min, max
  
  if (rangeString) {
    
    // Adjusted regex to capture floating point numbers.
    const match = rangeString.match(/\((\d+\.?\d*),\s*(\d+\.?\d*)\)/)
    
    // Used parseFloat() to handle floating-point numbers.
    if (match) {
      min = parseFloat(match[1])
      max = parseFloat(match[2])
    }
  }
  
  return (
    <>
      
      <Field
        className = "inputField"
        type = "number"
        id = {id}
        name = {name}
        step = "0.0001"
        style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black", borderColor: isDarkTheme ? "gray" : "black" } }
      />

      {/* Conditionally render the slider if min and max are available. */}
      { min !== undefined && max !== undefined && (
        <Field
          className = "sliderField"
          type = "range"
          id = { id }
          name = { name }
          min = { min }
          max = { max }
          step = "0.0001"
          style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black", borderColor: isDarkTheme ? "gray" : "black" } }
        />
      )}

    </>
  )

}

export default FloatField
