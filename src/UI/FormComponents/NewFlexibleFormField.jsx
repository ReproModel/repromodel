import "./Form.css"

import BooleanField from "../../components/ui/boolean-field"
import DefaultTextField from "../../components/ui/default-text-field"
import FormulaField from "../../components/ui/formula-field"
import React from "react"
import FloatField from "../../components/ui/float-field"
import SmartFreeTextField from "./FieldTypes/SmartStringField"
import IntegerField from "../../components/ui/integer-field"

import { useFormikContext } from "formik"

function FlexibleFormField({ id, label, type, name, object }) {
  
  const { setFieldValue, values } = useFormikContext()
  const defaultValue = object?.default
  
  // Set the default value when it's defined and the field value is not already set
  React.useEffect(() => {
    if (
      defaultValue !== undefined &&
      defaultValue !== null &&
      (values[name] === undefined || values[name] === null || values[name] === '')
    ) {
      setFieldValue(name, defaultValue)
    }
  }, [name, defaultValue, setFieldValue, values])

  const renderSwitch = () => {
    switch (type) {
      
      case "str":
        return <SmartFreeTextField id = { id } label = { label } name = { name } object = { object } />
      
      case "float":
        return <FloatField id = { id } label = { label } name = { name } object = { object } />
      
      case "int":
        return <IntegerField id = { id } label = { label } name = { name } object = { object } />
      
      case "slider":
        return <SliderField id = { id } label = { label } name = { name } />
      
      case "type(lambda x: x)":
        return <FormulaField id = { id } label = { label } name = { name } />
      
      case "bool":
        return <BooleanField id = { id } label = { label } name = { name } />
      
      default:
        return <DefaultTextField id = { id } label = { label } name = { name } type = { type } />
    }
  }

  return <>{ renderSwitch() }</>
}

export default FlexibleFormField