import React from "react"

import { Field } from "formik"

function FloatField({ id, label, object, name }) {
    
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
        />
      )}

    </>
  )

}

export default FloatField
