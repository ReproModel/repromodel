import { Field } from "formik";
import React from "react";
import "./Field.css";
import SmartIntegerField from "./FieldTypes/SmartIntegerField";
import SmartFreeTextField from "./FieldTypes/SmartStringField";
import FormulaField from "./FieldTypes/FormulaField";
import SmartFloatField from "./FieldTypes/SmartFloatField";

function FreeTextField({ id, label, name }) {
  return <Field className="inputField" id={id} name={name} label={label} />;
}

function SmartOldFloatField({ id, label, object }) {
  return (
    <Field
      className="inputField"
      type="number"
      id={id}
      name={label}
      step="0.01"
    />
  );
}



function IntegerField({ id, label, object }) {
  return (
    <>
      <Field
        className="inputField"
        type="number"
        id={id}
        name={label}
        step="1"
      />
      
    </>
  );
}


function SliderField({ id, label }) {
  return (
    <>
      <Field
        className="inputField"
        type="number"
        id={id}
        name={label}
        step="0.01"
      />
      <Field
        className="inputField"
        type="range"
        id={id}
        name={label}
        step="0.01"
      />
    </>
  );
}



function FlexibleFormField({
  id,
  label,
  type,
  object,
  name,
  options,
  moreQuestions,
}) {
  const renderSwitch = () => {
    switch (type) {
      case "str":
        return <SmartFreeTextField id={id} label={label} name={label} object={object}  />;
      case "float":
        return <SmartFloatField id={id} label={label} name={label} object={object} />;
      case "int":
        return (
          <SmartIntegerField id={id} label={label} name={label} object={object} />
        );
      case "slider":
        return <SliderField id={id} label={label} name={label} />;
      case "type(lambda x: x)":
        return <FormulaField id={id} label={label} name={label} />;
      default:
        return <div>Error occured, unsupported type</div>;
    }
  };
  return <>{renderSwitch()}</>;
}

export default FlexibleFormField;
