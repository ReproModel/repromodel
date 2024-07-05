import "./experiment-builder-wrapper.css"

import ExperimentBuilder from "./experiment-builder"
import ModalitySection from "../modality-section/modality-section"
import React from "react"

import { Form } from "formik"
import { useState } from "react"


function filterModels(jsonData, intersectedModels) {
  
  if (intersectedModels.length === 0) {
    return jsonData
  }

  const newModels = {}

  Object.entries(jsonData.models).forEach(([modelKey, modelValue]) => {
    
    Object.entries(modelValue).forEach(([innerModelKey, innerModelValue]) => {
      
      const fullKey = `${modelKey}>${innerModelKey}`
      
      if (intersectedModels.includes(fullKey)) {
        
        if (!newModels[modelKey]) {
          newModels[modelKey] = {}
        }
        
        newModels[modelKey][innerModelKey] = innerModelValue
      }
    })
  })

  return { ...jsonData, models: newModels }
}


const ExperimentBuilderWrapper = ({ FormikProps, handleFileChange, newQuestions }) => {
  
  const [filterChoosen, setFilterChoosen] = useState(false)
  const [selectedModels, setSelectedModels] = useState()
  
  return (
    <>
      { !filterChoosen && (
        <Form>
          <ModalitySection
            class_per_tag = { newQuestions.tags.class_per_tag }
            setFilterChoosen = { setFilterChoosen }
            setSelectedModels = { setSelectedModels }
          />
        </Form>
      )}

      { filterChoosen && (
        <ExperimentBuilder
          FormikProps = { FormikProps }
          setFieldValue = { FormikProps.setFieldValue }
          handleFileChange = { handleFileChange }
          newQuestions = { filterModels(newQuestions, selectedModels) }
        />
      )}
    </>
  )
}

export default ExperimentBuilderWrapper