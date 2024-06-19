import "./modality-options.css"

import React from 'react'

import { capitalizeFirstLetterOfEachWord } from "../../utils/string-helpers"
import { imageUrls } from './image-urls'

const ModalityOptions = ({ options, onOptionClick, selectedOptions, group}) => {
  
  return (
    
    <div className = "card-deck card-break">
      
      { options && (
        <>
          { Object.entries(options).map(([optionName, optionContent], idx) => (
            
            <div
              className = { `card ${selectedOptions.includes(optionName) ? 'selected' : ''}` }
              key = { optionName }
              onClick = { () => { onOptionClick(group, optionName) } }
            >
              
              <div className = "row">
                
                <div className = "card-image-container">
                  
                  <div
                    className = "card-image"
                    style = {{ backgroundImage: imageUrls[optionName] } }
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