import "./FormComponents/Form.css"

import NewFlexibleFormField from "./FormComponents/NewFlexibleFormField"
import React from "react"

import { Button, ButtonGroup, Typography } from "@mui/material"
import { capitalizeFirstLetter } from "../utils/string-helpers"
import { Field, Form, Formik } from "formik"
import { SmartFolderField } from "./FormComponents/FieldTypes/FolderQuestion"

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

const DynamicFormBuilder = ({ FormikProps, handleFileChange, newQuestions, setFieldValue }) => {
  
  return (
    <Form>
      
      {/* Optional JSON file upload input. */}
      <Typography>Optionally upload existing config file</Typography>
      
      <input
        type = "file"
        id = "uploadedJson"
        accept = ".json"
        onChange = { (event) => handleFileChange(event, setFieldValue) }
      />

      {/* Hidden field used to capture the submitType. */}
      <Field type = "hidden" name = "submitType"/>

      {/* Loop each folder. */}
      { Object.entries(newQuestions).map(([folder, folderContent]) => (
        
        <div style = { { display: "flex", flexDirection: "column" } }>
          
          <h4> { capitalizeFirstLetter(folder) }</h4>

          {/* Case 1: Folder is nested. */}
          { nestedFolders.includes(folder) ? (
            <>
              <SmartFolderField folder = { folder } folderContent = { folderContent } />

              {/* Loop each file in each folder. */}
              { Object.entries(folderContent).map(([file, fileContent]) => (
                <>
                  
                  {/* Loop each class in each file. */}
                  { Object.entries(fileContent).map(
                    
                    ([className, classContent]) => (
                      
                      <div style = { { paddingLeft: "16px" } }>
                        
                        { FormikProps.values[folder] &&
                          
                          ((Array.isArray(FormikProps.values[folder]) && FormikProps.values[folder].includes(className))
                            || (typeof FormikProps.values[folder] === "string" && FormikProps.values[folder] === className)) &&
                            
                            (
                              <div className = "paramBox">
                                
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
          
          // Case 2: Folder is flat.
          ): folder === "data_splits" ? (
            
            <div className = "paramBox">
              
              { Object.entries(folderContent).map(([param, value]) => (
                <>
                  
                  <label htmlFor = { `${folder}:${param}` }>{param}:</label>
                  
                  <NewFlexibleFormField
                    id = { `${folder}:${param}` }
                    object = { value }
                    type = { value.type }
                    name = { `${folder}:${param}` }
                    label = { param }
                  />   
                               
                </>
              ))}
            </div>
          ) : (
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
          )}

        </div>

      ))}

      <ButtonGroup variant = "outlined" sx = { { marginTop: "16px" } }>
        
        <Button
          type = "submit"
          onClick = { () => setFieldValue("submitType", "training") }>
          Submit for Training
        </Button>
        
        <Button
          type = "submit"
          onClick = { () => setFieldValue("submitType", "download") }>
          Download Config File
        </Button>
                  
      </ButtonGroup>

    </Form>
  )
}

export default DynamicFormBuilder