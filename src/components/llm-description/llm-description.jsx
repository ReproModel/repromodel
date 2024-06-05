import "./llm-description.css"

import React from "react"

import { Form } from "formik"
import { FormControl, FormControlLabel, FormLabel, Radio, RadioGroup, TextareaAutosize, Typography } from "@mui/material"

const LLMDescription = ({ FormikProps, handleFileChange, setFieldValue }) => {

  return (
    <Form>
        <Typography className = "heading">Generate methods section for your research paper.</Typography>


        <Typography className = "json-input-file-label">Upload existing configuration file.</Typography>
      
        <input
          type = "file"
          className = "json-input-file"
          accept = ".json"
          onChange = { (event) => handleFileChange(event, setFieldValue) }
        />

        <FormControl className = "textarea-additional-prompt">
          <FormLabel>Additional Prompting</FormLabel>
          <TextareaAutosize
            placeholder = "Additional rules..."
            minRows = { 4 }
            sx = { {
              '&::before': { display: 'none' },
              '&:focus-within': { outline: '2px solid var(--Textarea-focusedHighlight)', outlineOffset: '2px' }
            }}
          />
        </FormControl>

        <FormControl>
               
          <div className = "radio-writing-voice-container">
            
            <FormLabel className = "radio-writing-voice-label">Writing Voice</FormLabel>
            
            <RadioGroup className = "radio-writing-voice" row>
              
              <FormControlLabel value = "Passive Voice" control = { <Radio /> } label = "Passive Voice" />
              
              <FormControlLabel value = "Active Voice" control = { <Radio /> } label = "Active Voice" />
            
            </RadioGroup>
          
          </div>

          <div className = "radio-output-format-container">
            
            <FormLabel className = "radio-output-format-label">Output Format</FormLabel>
            
            <RadioGroup className = "radio-output-format" row>
              
              <FormControlLabel value = "LaTeX (.tex)" control = { <Radio /> } label = "LaTeX (.tex)" />
              
              <FormControlLabel value = "Plain Text (.txt)" control = { <Radio /> } label = "Plain Text (.txt)" className = "radio-plain-text" />
            
            </RadioGroup>
          
          </div>

      </FormControl>

    </Form>
  )
}

export default LLMDescription