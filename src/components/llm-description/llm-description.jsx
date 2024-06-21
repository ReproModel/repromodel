import { Form } from "formik"
import { Button, FormControlLabel, Radio, RadioGroup, TextareaAutosize, Typography } from "@mui/material"
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

// Function to be triggered when the config file is uploaded
export const handleFileUpload = (event, setConfig) => {
  const file = event.currentTarget.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
        setConfig(text)
    };
    reader.readAsText(file);
  }
};

const supportedLLMs = [
  { value: "Llama-3", label: <div><img src =  "./llm-models/meta.svg" height = "12px" width = "20px"/>{" "}Llama-3</div> }
]

const LLMDescription = () => {

  // Config File
  const [config, setConfig] = React.useState("")

  // LLM Model
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
    console.log("Config file contains: ", config)

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

      // Hide Loading Animation
      setLoading(false)

      // Set Form Elements
      setConfig(config)
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

      <Typography className = "heading">Generate the methods section for your research paper.</Typography>

      
      {/* Configuration File */}
      <div className = "json-input-file-container">

        {/* Subheader - Extract Code */}
        <div className = "json-input-file-label">
          <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>Upload Existing Configuration</span>
        </div>
        
        <input
                type = "file"
                className = "json-input-file"
                accept = ".json"
                onChange = { (event) => handleFileUpload(event, setConfig) }
              />


      </div>
      
      {/* Supported LLM Models */}
      <div className = "supported-llm-container">
        
        <div className = "supported-llm-label">
          <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>LLM Model</span>
        </div>

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
      <div className = "textarea-additional-prompt-container" style = { { width: "98%" } }>

        <div className = "textarea-additional-prompt-label">
            <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>Additional LLM Prompting Rules</span>
        </div>
          
        <TextareaAutosize
          className = "textarea-additional-prompt"
          onChange = { (e) => { setAdditionalPrompt(e.target.value) } }
          value = { additionalPrompt }
          placeholder = " Describe my custom model in more detail while only briefly describe other models that are used.&#10;&#10; Do not describe optimizers and LR schedulers."
          minRows = { 4 }
          style = { { width: "100%", height: "20px" } }
          sx = { {
            "&::before": { display: "none" },
            "&:focus-within": { outline: "2px solid var(--Textarea-focusedHighlight)", outlineOffset: "2px" }
          }}
        />
        
      </div>
      
      
      {/* Writing Voice */}
      <div className = "radio-writing-voice-container">
        
        <FormControlLabel label = {<span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>Writing Voice</span>} labelPlacement = "start" 

            control = {

            <RadioGroup id = "radio-writing-voice" className = "radio-writing-voice" onChange = { (e, value) => { setOutput(""); setVoice(value) }} sx = { { marginLeft: "50px", marginTop: "4px" } } row>

                <FormControlLabel value = "passive" control = { <Radio checked = { voice == "passive" } style = { { opacity: "50%", color: "black", fontSize: "12px", root: { "& .MuiSvgIcon-root": { height: "1.5em", width: "2.5em" } } } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Passive Voice</span> } />

                <FormControlLabel value = "active" control = { <Radio checked = { voice == "active" } style = { { opacity: "50%", color: "black", fontSize: "12px" } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Active Voice</span> } />

            </RadioGroup>

        }/>

      </div>



      {/* Output Format */}
      <div className = "radio-output-format-container">
        
        <FormControlLabel label = {<span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>Output Format</span>} labelPlacement = "start" 

            control = {

            <RadioGroup id = "radio-output-format" className = "radio-output-format" onChange = { (e, value) => { setOutput(""); setFormat(value) }} sx = { { marginLeft: "40px", marginTop: "4px" } } row>

                <FormControlLabel value = "latex" control = { <Radio checked = { format == "latex" } style = { { opacity: "50%", color: "black", fontSize: "12px", root: { "& .MuiSvgIcon-root": { height: "1.5em", width: "2.5em" } } } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>LaTeX (.tex)</span> } />

                <FormControlLabel value = "plain-text" control = { <Radio checked = { format == "plain-text" } style = { { opacity: "50%", color: "black", fontSize: "12px", marginLeft: "10px" } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Plain Text (.txt)</span> } />

            </RadioGroup>

        }/>

      </div>
      
       {/* Loading Animation */}
      { isLoading ?  <l-tailspin size = "40" stroke = "5" speed = "0.9"  color = "black" className = "loading-animation" /> :
        
        <div>

          {/* Button - Submit */}
          <div className = "copy-button">
            <Button variant = "contained" onClick = { () => runQuery(config, additionalPrompt, voice, format) }>
                <span style = { { marginTop: "4px", fontSize: "12px"} }>
                  Submit
                </span>
            </Button>
          </div>

          {/* LLM Response Output */}
          <div className = "textarea-llm-response-container" style = { { width: "100%" } }>
            
            <div className = "textarea-llm-response-label">
              <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px", fontWeight: "600" } }>LLM Response</span>
            </div>

            <div className = "container-content" style = { { width: "100%" } }>
              
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

          {/* Button - Copy Code */}
          <div className = "copy-button">
            <Button variant = "contained" onClick = { () => navigator.clipboard.writeText(output) }>
              <span style = { { marginTop: "4px", fontSize: "12px"} }>
                Copy
              </span>
            </Button>
          </div>

        </div>

    }

    </Form>
  )
}

export default LLMDescription