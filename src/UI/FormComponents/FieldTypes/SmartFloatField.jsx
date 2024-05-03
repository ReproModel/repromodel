import React from "react";
import { Field } from "formik";

function SmartFloatField({ id, label, object }) {
    // Extract and parse the range from the object, if it exists
    const rangeString = object?.range;  // Using optional chaining
    let min, max;
  
    if (rangeString) {
      // Adjusted regex to capture floating point numbers
      const match = rangeString.match(/\((\d+\.?\d*),\s*(\d+\.?\d*)\)/);
      if (match) {
        min = parseFloat(match[1]);  // Use parseFloat to handle floating-point numbers
        max = parseFloat(match[2]);
      }
    }
  
    return (
      <>
        <Field
          className="inputField"
          type="number"
          id={id}
          name={label}
          step="0.0001"
        />
        {/* Conditionally render the slider if min and max are available */}
        {min !== undefined && max !== undefined && (
          <Field
            className="sliderField"
            type="range"
            id={id}
            name={label}
            min={min}
            max={max}
            step="0.0001"
          />
        )}
      </>
    );
  }

  export default SmartFloatField;