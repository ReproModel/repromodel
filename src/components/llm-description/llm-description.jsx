import "./llm-description.css"

import CustomSelectComponent from "../ui/custom-select"
import React from "react"

import { Form } from "formik"
import { FormControlLabel, Radio, RadioGroup, TextareaAutosize, Typography } from "@mui/material"

const supportedLLMs = [
  { index: 0, label: "Llama-3-8b-Instruct", company: "Meta", image: "./llm-models/meta.svg" },
  { index: 1, label: "Llama-3-70b-Instruct", company: "Meta", image: "./llm-models/meta.svg" }
]

const LLMDescription = ({ handleFileChange, setFieldValue }) => {

  const runQuery = () => { 
    console.log("Running LLM query...")
  }

  return (
    <Form>
        
        <Typography className = "heading">Generate the methods section for your research paper.</Typography>

        
        {/* Configuration File */}
        <div className = "json-input-file-container">

          <Typography className = "json-input-file-label">Upload existing configuration file to pass to LLM.</Typography>
          
          <input
            type = "file"
            className = "json-input-file"
            accept = ".json"
            onChange = { (event) => handleFileChange(event, setFieldValue) }
          />

        </div>
        
        {/* Supported LLM Models */}
        <div className = "supported-llm-container">
          
          <Typography className = "supported-llm-label">LLM Model</Typography>

          <CustomSelectComponent
            className = "supported-llm"
            placeholder = "Select model..."
            isMulti = { false }
            options = { supportedLLMs }
          />

        </div>

        
        {/* Additional LLM Prompting Rules */}
        <div className = "textarea-additional-prompt-container" style = { { width: "100%" } }>

          <Typography className = "textarea-additional-prompt-label">Additional LLM Prompting Rules</Typography>
            
          <TextareaAutosize
            className = "textarea-additional-prompt"
            placeholder = ""
            minRows = { 4 }
            style = { { width: "100%", height: "40px" } }
            sx = { {
              '&::before': { display: 'none' },
              '&:focus-within': { outline: '2px solid var(--Textarea-focusedHighlight)', outlineOffset: '2px' }
            }}
          />
          
        </div>
        
        
        {/* Writing Voice */}
        <div className = "radio-writing-voice-container">
               
          <Typography className = "radio-writing-voice-label">Writing Voice</Typography>
            
          <RadioGroup className = "radio-writing-voice" row>
            
            <FormControlLabel value = "Passive Voice" control = { <Radio /> } label = "Passive Voice" />
            
            <FormControlLabel value = "Active Voice" control = { <Radio /> } label = "Active Voice" />
          
          </RadioGroup>

        </div>


        {/* Output Format */}
        <div className = "radio-output-format-container">
          
          <Typography className = "radio-output-format-label">Output Format</Typography>
            
          <RadioGroup className = "radio-output-format" row>
            
            <FormControlLabel value = "LaTeX (.tex)" control = { <Radio /> } label = "LaTeX (.tex)" />
            
            <FormControlLabel value = "Plain Text (.txt)" control = { <Radio /> } label = "Plain Text (.txt)" className = "radio-plain-text" />
          
          </RadioGroup>

        </div>

        {/* Submit Button */}
        <button className = "submit-button" onClick = { runQuery() }>
          <Typography>Submit</Typography>
        </button>

    </Form>
  )
}

export default LLMDescription