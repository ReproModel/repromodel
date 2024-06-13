import "./modality-options.css"

import React from 'react'

import { capitalizeFirstLetterOfEachWord } from "../../utils/string-helpers"

const ModalityOptions = ({ options, onOptionClick, selectedOptions, group}) => {
  
  return (
    
    <div className = "card-deck card-break">
      
      { options && (
        <>
          { Object.entries(options).map(([optionName, optionContent]) => (
            
            <div
              className = { `card ${selectedOptions.includes(optionName) ? 'selected' : ''}` }
              key = { optionName }
              onClick = { () => onOptionClick(group, optionName) }
            >
              
              <div className = "row">
                
                <div className = "card-image-container">
                  
                  <div
                    className = "card-image"
                    style = {{ backgroundImage: "url('https://production-media.paperswithcode.com/icons/task/48d55b59-3af2-4a6d-a195-572f1d4a1867.jpg')" } }
                  />
                
                </div>
                
                <div className = "card-title">
                  
                  <p>{ capitalizeFirstLetterOfEachWord(optionName) }</p>
                
                </div>
              
              </div>

            </div>

          ))}
        </>
      )}

    </div>
  )
}

export default ModalityOptions