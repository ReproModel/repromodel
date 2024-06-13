import "./experiment-builder-wrapper.css"

import ExperimentBuilder from "./experiment-builder"
import ModalitySection from "../modality-section/modality-section"
import React from "react"

import { Form } from "formik"
import { useState } from "react"

const ExperimentBuilderWrapper = ({ FormikProps, handleFileChange, newQuestions }) => {
  
  const [filterChoosen, setFilterChoosen] = useState(false)
  return (
    <>
      { !filterChoosen && (
        <Form>
          <>
            <ModalitySection tags = { newQuestions.tags.class_per_tag }/>

            <button type = "submit" className = "start-building-button" onClick = { () => setFilterChoosen(true) }>
              Start Building
            </button>
          </>
        
        </Form>
      )}
      
      { filterChoosen && (
        <ExperimentBuilder
          FormikProps = { FormikProps }
          setFieldValue = { FormikProps.setFieldValue }
          handleFileChange = { handleFileChange }
          newQuestions = { newQuestions }
        />
      )}

    </>
  )
}

export default ExperimentBuilderWrapper