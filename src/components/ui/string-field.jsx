import React from "react"

import { Field } from "formik"

function StringField({ id, label, object, name }) {

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])

  // Attempt to parse the options from the object, if they exist.
  const optionsArray = object?.options ? JSON.parse(object.options.replace(/'/g, '"')) : []

  return (
    <>
      
      { optionsArray.length > 0 ? (
        <Field className = "inputField" as = "select" id = { id } name = { name } style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black", borderColor: isDarkTheme ? "gray" : "black" } }>
          <>
            <option value = "">Select an option...</option>
            { optionsArray.map((option, index) => (
              <option key = { index } value = { option }>
                { option }
              </option>
            ))}
          </>
        </Field>
      ) : (
        <Field
          className = "inputField"
          type = "text"
          id = { id }
          name = { name }
          style = { { backgroundColor: isDarkTheme ? "gray" : "white", color:  isDarkTheme ? "white" : "black", borderColor: isDarkTheme ? "gray" : "black" } }
        />
      )}

    </>
  )
}
export default StringField