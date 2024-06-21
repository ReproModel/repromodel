import { capitalizeFirstLetter } from "../../utils/string-helpers"
import { Form } from "formik"
import { handleCustomScriptSubmit } from "../../utils/json-helpers"
import { handleDownload } from "../../utils/download-helpers"
import { highlight, languages } from "prismjs/components/prism-core"
import { Typography } from "@mui/material"

import axios from "axios"
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

  // Category
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
    <Form>

      <div className = "custom-script-heading">
        <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>What kind of custom script?</span>
      </div>


      {/* Supported Categories */}
      <div className = "category-container">
        <CustomSelectComponent
          className = "category-dropdown"
          isMulti = { false }
          options = { categories }
          value = { category }
          onChange = { (option) => { setCategory(option) } }
        />

        <button
          className = "load-template-button"
          onClick = { () => {
            fetchCustomScriptTemplate(category.value)
          }}
        >
          Load Template
        </button>
      </div>


      {/* Custom Script Editor */}
      <div className = "container-content" style = { { width: "100%" } }>
              
        <div className = "container-editor-area">
          
          <Editor
            className = "container-editor"
            value = { code }
            onValueChange = { (code) => setCode(code) }
            highlight = { (code) => highlight(code, languages.py) }
          />

        </div>

      </div>
        

      {/* Output File */}
      <label className = "file-name-input-container">
        Enter output file name:
        <input id = "file-name-input" className = "file-name-input" type = "text" placeholder = "Enter output file name (without .py)"/>
      </label>
        
      
      {/* Submit / Copy / Download Buttons */}
      <div className = "button-container">
          <button
            type = "submit"
            className = "button"
            onClick = { () => {

              const fileNameElement = document.getElementById("file-name-input")
              const fileNameValue = fileNameElement.value
              
              handleCustomScriptSubmit(code, fileNameValue, category.value)
            }}
          >
            Submit
          </button>

          <button type = "submit" className = "button right-button" onClick = { () => navigator.clipboard.writeText(code) }>
            Copy
          </button>

          <button
            type = "submit"
            className = "button right-button"
            onClick = { () => {
              handleDownload(code, `Custom${capitalizeFirstLetter(category.value)}Script`, "py")
            }}
          >
            Download
          </button>

      </div>

    </Form>
  )
}

export default CustomScript