import React from 'react'
import Select from 'react-select'

import { Field } from 'formik'

const onChange = (option) => {
  form.setFieldValue(
    field.name,
    isMulti ? option.map((item) => item.value) : option.value
  )
}

const getValue = (options, field, isMulti) => {
  if (options) {
    // Ensure field.value is an array and not null or undefined.
    const fieldValue = Array.isArray(field.value) ? field.value : []

    return isMulti
      ? options.filter(option => fieldValue.includes(option.value))
      : options.find(option => option.value === field.value)
  
  } else {
    return isMulti ? [] : ""
  }
}

const CustomSelectComponent = ({
  className,
  placeholder,
  field,
  form,
  options,
  isMulti = false,
  ...props
}) => {
  
  return (
    <Select
      className = { className }
      name = { field.name }
      value = { getValue(options, field, isMulti) }
      onChange = { onChange }
      placeholder = { placeholder }
      options = { options }
      isMulti = { isMulti }
      { ...props }
    />
  )
}

// Formik field wrapper for custom select.
const CustomSelect = (props) => (
  <Field component = { CustomSelectComponent } { ...props } />
)

export default CustomSelect
