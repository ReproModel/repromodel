import { Field } from "formik";
import React from "react";
import "./Field.css"

function FreeTextField({ id, label }) {
  return <Field className = "inputField" id={id} name={label} />;
}

function DecimalField({ id, label }) {
  return <Field className = "inputField"type="number" id={id} name={label} step="0.01" />;
}

function IntegerField({ id, label }) {
  return <Field className = "inputField" type="number" id={id} name={label} step="1" />;
}

function SliderField({ id, label }) {
  return (
    <>
      <Field className = "inputField"type="number" id={id} name={label}step="0.01" />
      <Field className = "inputField"type="range" id={id} name={label} step="0.01" />
    </>
  );
}

function FlexibleFormField({ id, label, type, options, moreQuestions }) {
  const renderSwitch = () => {
    switch (type) {
      case "text":
        return <FreeTextField id={id} label={label} />;
      case "decimal":
        return <DecimalField id={id} label={label} />;
      case "integer":
        return <IntegerField id={id} label={label} />;
      case "slider":
        return <SliderField id={id} label={label} />;
      default:
        return <div>Error occured, unsupported type</div>;
    }
  };
  return <>{renderSwitch()}</>;
}

export default FlexibleFormField;
