import React from "react";
import { Field } from "formik";

function SmartIntegerField({ id, label, object, name}) {
    // Extract and parse the range from the object, if it exists
    const rangeString = object?.range;  // Using optional chaining
    let min, max;
  
    if (rangeString) {
      const match = rangeString.match(/\((\d+),\s*(\d+)\)/);
      if (match) {
        min = parseInt(match[1], 10);
        max = parseInt(match[2], 10);
      }
    }
  
    return (
      <>
        <Field
          className="inputField"
          type="number"
          id={id}
          name={name}
          step="1"
        />
        {/* Conditionally render the slider if min and max are available */}
        {min !== undefined && max !== undefined && (
          <Field
            className="sliderField"
            type="range"
            id={id}
            name={name}
            min={min}
            max={max}
            step="1"
          />
        )}
      </>
    );
  }

  export default SmartIntegerField;