import React from "react";
import { Field } from "formik";

function SmartFreeTextField({ id, label, object }) {
  // Attempt to parse the options from the object, if they exist
  const optionsArray = object?.options
    ? JSON.parse(object.options.replace(/'/g, '"'))
    : [];

  return (
    <>
      {optionsArray.length > 0 ? (
        <Field as="select" className="inputField" id={id} name={label}>
          <>
            <option value="">Select an option</option>
            {optionsArray.map((option, index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </>
        </Field>
      ) : (
        <Field
          className="inputField"
          type="text"
          id={id}
          name={label}
          placeholder="not a dropdown"
        />
      )}
    </>
  );
}
export default SmartFreeTextField;
