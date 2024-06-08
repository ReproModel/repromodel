import { Form } from "formik"
import { FormControlLabel, Radio, RadioGroup, TextareaAutosize, Typography } from "@mui/material"
import { highlight, languages } from "prismjs/components/prism-core"
import { tailspin } from 'ldrs'

import CustomSelectComponent from "../ui/custom-select"
import Editor from "react-simple-code-editor"
import React from "react"
import axios from "axios"

import "./llm-description.css"
import "prismjs/components/prism-clike"
import "prismjs/components/prism-latex"
import "prismjs/themes/prism.css"

const supportedLLMs = [
  { value: "Llama-3-70b", label: <div><img src =  "./llm-models/meta.svg" height = "12px" width = "20px"/>{" "}Llama-3-70b</div> },
  { value: "Llama-3-8b", label: <div><img src =  "./llm-models/meta.svg" height = "12px" width = "20px"/>{" "}Llama-3-8b</div> },
  { value: "Mistral-7b", label: <div><img src =  "./llm-models/mistral.svg" height = "12px" width = "20px"/>{" "}Mistral-7b</div> }
]

const LLMDescription = ({ handleFileChange, setFieldValue }) => {

  // Config
  const [LLM, setLLM] = React.useState(supportedLLMs[0])

  // LLM Additional Prompt
  const [additionalPrompt, setAdditionalPrompt] = React.useState("")

  // Radio Button - Writing Voice
  const [voice, setVoice] = React.useState("passive")
  
  // Radio Button - Output File Format
  const [format, setFormat] = React.useState("latex")

  // LLM Output
  const [output, setOutput] = React.useState("")

  // Loading Animation
  const [isLoading, setLoading] = React.useState(false)
  tailspin.register()

  const runQuery = async (config, additionalPrompt, voice, format) => { 

    console.log("Running LLM query...")

    console.log("Text Area - Additional Prompt: ", additionalPrompt)
    console.log("Radio Radio - Writing Voice: ", voice)
    console.log("Radio Radio - Output File Format: ", format)

    const formData = new FormData();
    formData.append('config', config);
    formData.append('additionalPrompt', additionalPrompt);
    formData.append('voice', voice);
    formData.append('format', format);
    
    setLoading(true)

    const response = await axios.post('http://127.0.0.1:5005/generate_methodology', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then((response) => {
      console.log(response.data)

      console.log(formData)

      // Hide Loading Animation
      setLoading(false)

      // Set Form Elements
      setAdditionalPrompt(additionalPrompt)
      setVoice(voice)
      setFormat(format)

      // Display LLM Output
      setOutput(response.data)
      
      return response.data.methodology
    })
    .catch((error) => {
      console.error('Error generating methodology:', error)
      throw error
    })
  }

  return (
    <Form>

        { isLoading ?  <l-tailspin size = "40" stroke = "5" speed = "0.9"  color = "black" className = "loading-animation" /> :
        
          <div>
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
                value = { LLM }
                onChange = { (option) => { setLLM(option) } }
              />

            </div>

            
            {/* Additional LLM Prompting Rules */}
            <div className = "textarea-additional-prompt-container" style = { { width: "100%" } }>

              <Typography className = "textarea-additional-prompt-label">Additional LLM Prompting Rules</Typography>
                
              <TextareaAutosize
                className = "textarea-additional-prompt"
                onChange = { (e) => { setAdditionalPrompt(e.target.value) } }
                value = { additionalPrompt }
                placeholder = "• Describe my custom model in more detail while only briefly describe other models that are used.&#10;&#10;• Do not describe optimizers and LR schedulers."
                minRows = { 4 }
                style = { { width: "100%", height: "40px" } }
                sx = { {
                  "&::before": { display: "none" },
                  "&:focus-within": { outline: "2px solid var(--Textarea-focusedHighlight)", outlineOffset: "2px" }
                }}
              />
              
            </div>
            
            
            {/* Writing Voice */}
            <div className = "radio-writing-voice-container">
                  
              <Typography className = "radio-writing-voice-label">Writing Voice</Typography>
                
              <RadioGroup className = "radio-writing-voice" onChange = { (e, value) => { setOutput(""); setVoice(value); }} row>
                
                <FormControlLabel value = "passive" control = { <Radio checked = { voice == "passive" }/> } label = "Passive Voice" />
                
                <FormControlLabel value = "active" control = { <Radio checked = { voice == "active" }/> } label = "Active Voice" />
              
              </RadioGroup>

            </div>


            {/* Output Format */}
            <div className = "radio-output-format-container">
              
              <Typography className = "radio-output-format-label">Output Format</Typography>
                
              <RadioGroup className = "radio-output-format" onChange = { (e, value) => { setOutput(""); setFormat(value); }} row>
                
                <FormControlLabel value = "latex" control = { <Radio checked = { format == "latex" }/> } label = "LaTeX (.tex)" />
                
                <FormControlLabel value = "plain-text" control = { <Radio checked = { format == "plain-text" }/> } label = "Plain Text (.txt)" className = "radio-plain-text" />
              
              </RadioGroup>

            </div>

            {/* Submit Button */}
            <button className = "submit-button" type="button" onClick = { () => runQuery("Method is bruteforce", additionalPrompt, voice, format) }>
              <Typography>Submit</Typography>
            </button>

            {/* LLM Response Output */}
            <div className = "textarea-llm-response-container" style = { { width: "100%" } }>

              <Typography className = "textarea-llm-response-label">LLM Response</Typography>

              <div className = "container-content" style = {{width: "100%"}}>
                <div className = "container-editor-area">
                  <Editor
                    className = "container-editor"
                    value = { output }
                    onValueChange = { (output) => setOutput(output) }
                    highlight = { (output) => (format == "latex") ? highlight(output, languages.latex) : highlight(output, languages.text) }
                  />
                </div>
              </div>
            </div>

            {/* Copy Button */}
            <button type = "submit" className = "copy-button" onClick = { () => navigator.clipboard.writeText(code) }>
              Copy
            </button>
            
          </div>  
        
        }

    </Form>
  )
}

export default LLMDescription