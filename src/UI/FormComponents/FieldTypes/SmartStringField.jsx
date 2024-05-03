import React from "react";
import { Field } from "formik";


function SmartFreeTextField({ id, label, object }) {
  // Attempt to parse the options from the object, if they exist
  let options = [];
  if (object?.options) {
    try {
      // Assume the options string is a JSON-like array string and parse it
      options = JSON.parse(object.options.replace(/'/g, '"')); // Replace single quotes to double quotes for valid JSON
    } catch (error) {
      console.error("Failed to parse options:", error);
      options = [];
    }
  }

  return (
    <>
      {options.length > 0 ? (
        <Field as="select" className="inputField" id={id} name={label}>
          <>
            <option value="">Select an option</option>
            {JSON.parse(options.replace(/'/g, '"')).map((option, index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </>
        </Field>
      ) : (
        <Field className="inputField" type="text" id={id} name={label} />
      )}
    </>
  );
}
export default SmartFreeTextField;