import "./llm-description.css"

import React from "react"

import { Form } from "formik"
import { Typography } from "@mui/material"

const LLMDescription = ({ FormikProps, handleFileChange, setFieldValue }) => {

  return (
    <Form>
        <Typography className = "json-input-file-label">Upload existing configuration file.</Typography>
      
        <input
            type = "file"
            className = "json-input-file"
            accept = ".json"
            onChange = { (event) => handleFileChange(event, setFieldValue) }
        />
    </Form>
  )
}

export default LLMDescription