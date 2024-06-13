import "./modality-options.css"

import ModalityOptions from "./modality-options"
import React from "react"

import { capitalizeFirstLetter } from "../../utils/string-helpers"
import { Typography } from "@mui/material"
import { useState } from "react"

///////////////////////////////////////////////////////////
// getModels()
///////////////////////////////////////////////////////////

function getModels(selectedOptions, jsonData) {
  
  const { task, subtask, modality, submodality } = selectedOptions
  const data = jsonData

  // Helper function to get models for a given key and values.
  function getModelsForKey(key, values) {
    
    // To be used dynamically in the future.
    const typeOfWhatIWantToGet = "models"
    
    let models = []
    
    if (data.tags[key] && data.tags[key][values] && data.tags[key][values][typeOfWhatIWantToGet]) {
      models = data.tags[key][values][typeOfWhatIWantToGet]
    }

    return models
  }

  // Get models for each selected option.
  const taskModels = getModelsForKey("task", task)
  const subtaskModels = getModelsForKey("subtask", subtask)
  const modalityModels = getModelsForKey("modality", modality)
  const submodalityModels = getModelsForKey("submodality", submodality)

  console.log("Task Models:", taskModels)
  console.log("Subtask Models:", subtaskModels)
  console.log("Modality Models:", modalityModels)
  console.log("Submodality Models:", submodalityModels)

  // Intersect all model arrays.
  function intersect(arr1, arr2) {
    if (arr1.length === 0) return arr2
    if (arr2.length === 0) return arr1
    return arr1.filter((value) => arr2.includes(value))
  }

  let intersectedModels = taskModels

  if (subtaskModels.length > 0) {
    intersectedModels = intersect(intersectedModels, subtaskModels)
  }
  if (modalityModels.length > 0) {
    intersectedModels = intersect(intersectedModels, modalityModels)
  }
  if (submodalityModels.length > 0) {
    intersectedModels = intersect(intersectedModels, submodalityModels)
  }

  return intersectedModels
}


///////////////////////////////////////////////////////////
// filterModels()
///////////////////////////////////////////////////////////

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


///////////////////////////////////////////////////////////
// ModalitySection component
///////////////////////////////////////////////////////////

const ModalitySection = (class_per_tag) => {
  
  const [selectedOptions, setSelectedOptions] = useState({
    task: [],
    subtask: [],
    modality: [],
    submodality: []
  })

  const handleOptionClick = (group, option) => {
    
    setSelectedOptions((prevSelectedOptions) => {
      
      const groupOptions = prevSelectedOptions[group]

      // Check if the option is already selected for the group.
      if (groupOptions && groupOptions.includes(option)) {

        // Deselect the option if it is already selected.
        return { ...prevSelectedOptions, [group]: [] }
      
      } else {
        // Select the new option, replacing any previously selected option.
        return { ...prevSelectedOptions, [group]: [option] }
      }
    })
  }

  const selectedModels = getModels(selectedOptions, class_per_tag)

  return (
    <div className = "container">
      
      { Object.entries(class_per_tag).map(([tag, tagContent]) => (
        
        <>
          
          { Object.entries(tagContent).map(([category, options]) => (
            
            <>
              <Typography style = { { marginTop: "16px", marginBottom: "4px" } } variant = "h7">
                Choose { capitalizeFirstLetter(category) }
              </Typography>
              
              <ModalityOptions
                options = { options }
                onOptionClick = { handleOptionClick }
                selectedOptions = { selectedOptions[category] || [] }
                group = { category }
              />
            </>

          ))}

        </>

      ))}
      
      { console.log("The selected models are: ", selectedModels) }

    </div>
  )
}

export default ModalitySection