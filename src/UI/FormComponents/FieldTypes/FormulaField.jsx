import React from "react";
import { Field } from "formik";

function FormulaField({ id, label, name }) {
    return (
      <Field
        className="inputField"
        id={id}
        name={name}
        label={label}
        placeholder="Enter your Formula"
      />
    );
  }

  export default FormulaField;