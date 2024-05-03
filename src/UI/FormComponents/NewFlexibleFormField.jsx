import { Field } from "formik";
import React from "react";
import "./Field.css";
import SmartIntegerField from "./FieldTypes/SmartIntegerField";
import SmartFreeTextField from "./FieldTypes/SmartStringField";
import FormulaField from "./FieldTypes/FormulaField";
import SmartFloatField from "./FieldTypes/SmartFloatField";

function DefaultTextField({ id, label, name }) {
  return (
    <Field
      className="inputField"
      id={id}
      name={name}
      label={label}
      placeholder={`Please enter ${type}`}
    />
  );
}

function FlexibleFormField({ id, label, type, object }) {
  const renderSwitch = () => {
    switch (type) {
      case "str":
        return (
          <SmartFreeTextField
            id={id}
            label={label}
            name={label}
            object={object}
          />
        );
      case "float":
        return (
          <SmartFloatField id={id} label={label} name={label} object={object} />
        );
      case "int":
        return (
          <SmartIntegerField
            id={id}
            label={label}
            name={label}
            object={object}
          />
        );
      case "slider":
        return <SliderField id={id} label={label} name={label} />;
      case "type(lambda x: x)":
        return <FormulaField id={id} label={label} name={label} />;
      default:
        return (
          <DefaultTextField id={id} label={label} name={label} type={type} />
        );
    }
  };
  return <>{renderSwitch()}</>;
}

export default FlexibleFormField;
