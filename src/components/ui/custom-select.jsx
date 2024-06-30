import React from 'react'
import Select from 'react-select'

import { Field } from 'formik'

const CustomSelectComponent = ({
  className,
  placeholder,
  field,
  form,
  options,
  isMulti = false,
  ...props
}) => {
  
  const onChange = (option) => {
    form.setFieldValue(
      field.name,
      isMulti ? option.map((item) => item.value) : option.value
    )
  }

  const getValue = () => {
    if (options) {
      
      const fieldValue = Array.isArray(field.value) ? field.value : []
      return isMulti ? options.filter(option => fieldValue.includes(option.value)) : options.find(option => option.value === field.value)
   
    } else {
      return isMulti ? [] : ""
    }
  }

  return (
    <Select
      className = { className }
      name = { field.name }
      value = { getValue() }
      onChange = { onChange }
      placeholder = { placeholder }
      options = { options }
      isMulti = { isMulti }
      { ...props }
      styles={{
        control: (baseStyles, state) => ({
          ...baseStyles,
          backgroundColor: "gray",
          borderColor: "gray",
          "::placeholder": 'white'
        }),
      }}
    />
  )
}

// Formik field wrapper for custom select.
const CustomSelect = (props) => (
  <Field component = { CustomSelectComponent } { ...props } />
)

export default CustomSelect