import { Form } from "formik"
import { handleCustomScriptSubmit } from "../../utils/json-helpers"
import { highlight, languages } from "prismjs/components/prism-core"

import axios from "axios"
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import CustomSelectComponent from "../ui/custom-select"
import dedent from "dedent"
import Editor from "react-simple-code-editor"
import React from "react"

import "./custom-script.css"
import "prismjs/components/prism-clike"
import "prismjs/components/prism-python"
import "prismjs/themes/prism.css"

const categories = [
  { value: "augmentations", label: "augmentations" },
  { value: "datasets", label: "datasets" },
  { value: "early_stopping", label: "early_stopping" },
  { value: "losses", label: "losses" },
  { value: "metrics", label: "metrics" },
  { value: "models", label: "models" },
  { value: "postprocessing", label: "postprocessing" },
  { value: "preprocessing", label: "preprocessing" }
]

const CustomScript = ({}) => {

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])

  const [category, setCategory] = React.useState(categories[0])

  const [code, setCode] = React.useState(
    dedent`
    Select the kind of custom script you want to create. 
    Make sure you have the backend running. 
    `
  )

  const fetchCustomScriptTemplate = async (type) => {
    try {
      const response = await axios.get(
        "http://127.0.0.1:5005/get-custom-script-template",
        { params: { type } }
      )
      setCode(response.data)
    } catch (error) {
      console.error("Error fetching data: ", error);
    }
  }

  return (
    <Form className = "custom-script-container">

      {/* Category Label */}
      <div 
        className="custom-script-category-label" 
        style={{ fontSize: 'calc(18px + 0.75vw)' }}
      >
        What kind of custom script?
      </div>

      <div className = "custom-script-category-container">
        
        {/* Category Dropdown */}
        <CustomSelectComponent
          className = "custom-script-category-dropdown"
          isMulti = { false }
          options = { categories }
          value = { category }
          onChange = { (option) => { setCategory(option) } }
        />

        {/* Load Template Button */}
        <button className = "custom-script-load-template-button" type = "button" onClick = { () => { fetchCustomScriptTemplate(category.value) } }>
          Load Template
        </button>

      </div>


      {/* Custom Script Editor */}
      <div className = "custom-script-container-content">
              
        <div className = "custom-script-container-editor-area">
          
          {/* Copy Button */}
          <div title = "Copy">
            <ContentCopyIcon
              onClick = { () => navigator.clipboard.writeText(code) }
              sx = { { "&:hover": { opacity: "30%" }, opacity: "50%", position: "absolute", top: "10px", right: "15px", zIndex: "100", cursor: "pointer", color: (isDarkTheme ? "white" : "black") } }
            />
          </div>

          {/* Editor */}
          <Editor
            className = "custom-script-container-editor"
            highlight = { (code) => highlight(code, languages.py) }
            onValueChange = { (code) => setCode(code) }
            value = { code }
          />

        </div>

      </div>        
      
      <div className = "custom-script-save-container">

          {/* Output File Name */}
          <input id = "custom-script-file-name-input" className = "custom-script-file-name-input" type = "text" placeholder = "File Name (without .py)"/>
          
          {/* Save Buttons */}
          <button
            className = "custom-script-save-button"
            type = "button"
            onClick = { () => {
              const fileNameValue = document.getElementById("custom-script-file-name-input").value              
              handleCustomScriptSubmit(code, fileNameValue, category.value)
            }}
          >
            Save
          </button>

      </div>

    </Form>
  )
}

export default CustomScript