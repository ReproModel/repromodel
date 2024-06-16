import "./model-testing.css"

import FlexibleFormField from "../ui/flexible-form-field/flexible-form-field"
import StopIcon from '@mui/icons-material/Stop'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import React from "react"

import { Button, ButtonGroup } from "@mui/material"
import { capitalizeFirstLetter } from "../../utils/string-helpers"
import { Field, Form } from "formik"
import { FolderField } from "../ui/folder-field"
import { Typography } from "@mui/material"

const foldersPartOfTesting = [
  "models",
  "datasets",
  "metrics",
  "device"
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
  "early_stopping"
]

const ModelTesting = ({ FormikProps, handleFileChange, newQuestions, setFieldValue }) => {

  const [testingInProgress, setTestingInProgress] = React.useState(false)
  
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
                  <FolderField folder = { folder } folderContent = { folderContent } />

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
                                        
                                        <FlexibleFormField
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
                  
                  <FlexibleFormField
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
  

  <ButtonGroup variant = "outlined" sx = { { marginTop: "16px" } }>
        
        {/* Start Testing Button */}
        { testingInProgress == false &&
            <Button
              type = "submit"
              style = { { width: "170px" } }
              onClick = { () => { setTestingInProgress(true); setFieldValue("submitType", "testing") } }
            >
              <PlayArrowIcon />
              Test
            </Button>
        
        }

        {/* Stop Testing Button */}
        { testingInProgress == true &&
            <Button
              type = "submit"
              style = { { width: "170px" } }
              onClick = { () => { setTestingInProgress(false); setFieldValue("submitType", "stop-testing") } }
            >
              <StopIcon/>
              Stop Testing
            </Button>
        }

      </ButtonGroup>

    </Form>
  )
}

export default ModelTesting