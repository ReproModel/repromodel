import { capitalizeFirstLetter } from "../../../utils/string-helpers"
import { handleCustomScriptSubmit } from "../../../utils/json-helpers"
import { handleDownload } from "../../../utils/download-helpers"
import { highlight, languages } from "prismjs/components/prism-core"

import dedent from "dedent"
import Editor from "react-simple-code-editor"
import React from "react"
import axios from "axios"

import "./code-editor.css"
import "prismjs/components/prism-clike"
import "prismjs/components/prism-python"
import "prismjs/themes/prism.css"

const CodeEditor = ({ label }) => {
  
  const [code, setCode] = React.useState(
    dedent`
    Select the kind of custom script you want to create. 
    Make sure to have the backend up and running. 
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
    <div>
      
      <div className = "container">
        
        <div className = "category-container">
          
          <label className = "category-lable" htmlFor = "category">What kind of custom script?</label>

          <select id = "category-dropdown" className = "category-dropdown" name = "category">
            <option value="augmentations">augmentations</option>
            <option value="datasets">datasets</option>
            <option value="early_stopping">early_stopping</option>
            <option value="losses">losses</option>
            <option value="metrics">metrics</option>
            <option value="models">models</option>
            <option value="postprocessing">postprocessing</option>
            <option value="preprocessing">preprocessing</option>
          </select>

          <button
            className = "load-template-button"
            onClick = { () => {
              const categoryElement = document.getElementById("category-dropdown")
              const categoryValue = categoryElement.value
              fetchCustomScriptTemplate(categoryValue)
            }}
          >
            Load Template
          </button>
        </div>

        <div className = "container-content">
          <div className = "container-editor-area">
            <Editor
              className = "container-editor"
              value = { code }
              onValueChange = { (code) => setCode(code) }
              highlight = { (code) => highlight(code, languages.py) }
            />
          </div>
        </div>

        <label className = "file-name-input-container">
          Enter file name:
          <input id = "file-name-input" className = "file-name-input" type = "text" placeholder = "Enter file name (without .py)"/>
        </label>

        <div className = "button-container">
          <button
            type = "submit"
            className = "button"
            onClick = { () => {

              const fileNameElement = document.getElementById("file-name-input")
              const fileNameValue = fileNameElement.value

              const categoryElement = document.getElementById("category-dropdown")
              const categoryValue = categoryElement.value
              
              handleCustomScriptSubmit(code, fileNameValue, categoryValue)
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
              const categoryElement = document.getElementById("category-dropdown")
              const categoryValue = categoryElement.value;
              handleDownload(code, `Custom${capitalizeFirstLetter(categoryValue)}Script`, "py")
            }}
          >
            Download
          </button>

        </div>

      </div>

    </div>

  )
}

export default CodeEditor