import React from 'react'
import Select from 'react-select'

import { Field } from 'formik'

const CustomSelectComponent = ({ className, placeholder, field, form, options, isMulti = false, ...props }) => {

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])
  
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
          backgroundColor: isDarkTheme ? "gray" : "white",
          borderColor: "gray",
          "::placeholder": isDarkTheme ? "white" : "black"
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