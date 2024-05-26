import React from "react"

import { Field } from "formik"

function SmartIntegerField({ id, label, object, name}) {

    // Extract and parse the range from the object, if it exists by using optional chaining.
    const rangeString = object?.range
    
    let min, max
  
    try {
      
      if (rangeString) {
        
        // Adjusted regex to capture floating point numbers.
        const match = rangeString.match(/\((\d+),\s*(\d+)\)/)
        
        // Use parseFloat() to handle floating-point numbers.
        if (match) {
          min = parseInt(match[1], 10)
          max = parseInt(match[2], 10)
        }
      }
  
    } catch (error) {
      console.error('An error occurred while parsing the range:', error)
    }
  
    return (
      <>
        <Field
          className = "inputField"
          type = "number"
          id = { id }
          name = { name }
          step = "1"
        />
        
        {/* Conditionally render the slider if min and max are available. */}
        {min !== undefined && max !== undefined && (
          <Field
            className = "sliderField"
            type = "range"
            id = { id }
            label = { label }
            name = { name }
            min = { min }
            max = { max }
            step = "1"
          />
        )}
      </>
    )
  }

  export default SmartIntegerField