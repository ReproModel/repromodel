import React from "react"

import { Field } from "formik"

function IntegerField({ id, label, object, name}) {

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])
    
  // Extract and parse the range from the object, if it exists by using optional chaining.
  const rangeString = object?.range
  let min, max

  try {
    
    if (rangeString) {
      
      const match = rangeString.match(/\((\d+),\s*(\d+)\)/)
      
      if (match) {
        min = parseInt(match[1], 10)
        max = parseInt(match[2], 10)
      }
    }

  } catch (error) {
    console.error('An error occurred while parsing the range: ', error)
  }
  
  
    return (
      <>
        
        <Field
          className = "inputField"
          type = "number"
          id = { id }
          name = { name }
          step = "1"
          style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black", borderColor: isDarkTheme ? "gray" : "black" } }
        />

        {/* Conditionally render the slider if min and max are available */}
        { min !== undefined && max !== undefined && (
          <Field
            className = "sliderField"
            type = "range"
            id = { id }
            label = { label }
            name = { name }
            min = { min }
            max = { max }
            step = "1"
            style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black", borderColor: isDarkTheme ? "gray" : "black" } }
          />
        )}

      </>
    )
  }

  export default IntegerField