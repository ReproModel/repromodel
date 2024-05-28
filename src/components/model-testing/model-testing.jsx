import "./model-testing.css"

import NewFlexibleFormField from "../../UI/FormComponents/NewFlexibleFormField"
import React from "react"

import { capitalizeFirstLetter } from "../../utils/string-helpers"
import { Field, Form, Formik } from "formik"
import { SmartFolderField } from "../../UI/FormComponents/FieldTypes/SmartFolderField"
import { Typography } from "@mui/material"

const foldersPartOfTesting = [
  "models",
  "datasets",
  "metrics",
  "device",
]

const nestedFolders = [
  "models",
  "datasets",
  "preprocessing",
  "postprocessing",
  "metrics",
  "losses",
  "augmentations",
  "lr_schedulers",
  "optimizers",
  "early_stopping",
]

const ModelTesting = ({ FormikProps, handleFileChange, newQuestions, setFieldValue }) => {
  
  return (
    <Form>
      
      {/* Optional JSON file upload input. */}
      <Typography className = "json-input-file-label">Upload existing training configuration file and reduce the selections if desired.</Typography>
      
      <input
        type = "file"
        id = "uploadedJson"
        className = "json-input-file"
        accept = ".json"
        onChange = { (event) => handleFileChange(event, setFieldValue) }
      />

      {/* Hidden field used to capture the submitType. */}
      <Field type = "hidden" name = "submitType" />
      
      {/* Loop each folder. */}
      { Object.entries(newQuestions).map(([folder, folderContent]) => (
        <>
          { foldersPartOfTesting.includes(folder) && (
            
            <div style = { { display: "flex", flexDirection: "column" } }>
              
              <h4> {capitalizeFirstLetter(folder)}</h4>

              {/* Case 1: Folder is nested and part of testing. */}
              { foldersPartOfTesting.includes(folder) && nestedFolders.includes(folder) ? (
                
                <>
                  <SmartFolderField folder = { folder } folderContent = { folderContent } />

                  {/* Loop each file in each folder. */}
                  { Object.entries(folderContent).map(([file, fileContent]) => ( 
                    <>
                      
                      {/* Loop each class in each file. */}
                      { Object.entries(fileContent).map(
                        
                        ([className, classContent]) => (
                          
                          <div style = { { paddingLeft: "16px" } }>
                            
                            { FormikProps.values[folder] && folder === "metrics" &&
                              ((Array.isArray(FormikProps.values[folder]) && FormikProps.values[folder].includes(className))
                              || (typeof FormikProps.values[folder] === "string" && FormikProps.values[folder] === className)) &&
                              
                              (
                                <div className = "param-box">
                                  
                                  <p>{ className } Params</p>

                                  {/* Conditionally render param questions if the class is selected. */}
                                  { Object.entries(classContent).map(
                                    ([param, value]) => (
                                      <>
                                        
                                        <label htmlFor = { `${folder}_params:${className}:${param}` }>
                                          { param }:
                                        </label>
                                        
                                        <NewFlexibleFormField
                                          id = { `${folder}_params:${className}:${param}` }
                                          object = { value }
                                          type = { value.type }
                                          name = { `${folder}_params:${className}:${param}` }
                                          label = { param }
                                        />

                                      </>
                                    )
                                  )}

                                </div>

                              )}

                          </div>

                        )

                      )}

                    </>
                  ))}
                </>

              // Case 2: Folder is flat and part of testing.
              ): foldersPartOfTesting.includes(folder) ? (
                <>
                  
                  <label htmlFor = { `${folder}` }>{ folder }:</label>
                  
                  <NewFlexibleFormField
                    id = { `${folder}` }
                    object = { folderContent }
                    type = { folderContent.type }
                    name = { `${folder}` }
                    label = { folder }
                  />

                </>
              ) : (
                <></>
              )}

            </div>

          )}

        </>

      ))}
  

      <button type = "submit" className = "submit-button" onClick = { () => setFieldValue("submitType", "testing") }>
        <Typography>Submit For Testing</Typography>
      </button>

    </Form>
  )
}

export default ModelTesting