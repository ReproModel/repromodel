import "./model-testing.css"

import axios from "axios"
import FlexibleFormField from "../ui/flexible-form-field/flexible-form-field"
import StopIcon from '@mui/icons-material/Stop'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import React from "react"

import { Button, ButtonGroup } from "@mui/material"
import { capitalizeFirstLetter } from "../../utils/string-helpers"
import { Field, Form } from "formik"
import { FolderField } from "../ui/folder-field"

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

  React.useEffect(() => {
    
    const interval = setInterval(() => {
      
      axios.get("http://127.0.0.1:5005/ping")
        
        .then(response => {
          if (response.data.testingInProgress === true) {
            setTestingInProgress(true)
          } else {
            setTestingInProgress(false)
          }
        })      

        .catch(error => {
          setTestingInProgress(false)
        })

    }, 3000)

    return () => clearInterval(interval)

  }, [])
  
  return (
    <Form>
      
      {/* Optional JSON file upload input. */}
      <div style = { { marginLeft: "96px" } }>
        <div className = "json-input-file-label">
          <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "18px", fontWeight: "700" } }>Upload Existing Testing Configuration</span>
        </div>

        <input
          type = "file"
          id = "uploadedJson"
          className = "json-input-file"
          accept = ".json"
          onChange = { (event) => handleFileChange(event, setFieldValue) }
        />
      </div>

      {/* Hidden field used to capture the submitType. */}
      <Field type = "hidden" name = "submitType" />
      
      {/* Loop each folder. */}
      { Object.entries(newQuestions).map(([folder, folderContent]) => (
        <div style = { { marginTop: "12px", paddingLeft: "96px" } }>
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

        </div>

      ))}
  

      <ButtonGroup variant = "outlined" sx = { { marginTop: "36px", marginLeft: "96px" } }>
        
        {/* Start Testing Button */}
        
        { !testingInProgress &&
          <div>
            <Button type = "submit" onClick = { () => { setFieldValue("submitType", "testing") } } style = { { width: "220px", backgroundColor: "#38512f", borderColor: "#38512f", color: "white", opacity: "90%" } }>
              <PlayArrowIcon style = { { fontSize: "14px" } }/>
              <span style = { { marginTop: "4px", marginLeft: "12px", marginRight: "12px", fontSize: "12px", textAlign: "center"} }>
                Test
              </span>
            </Button>
          </div>
        }

        {/* Stop Testing Button */}
        { testingInProgress &&
          <div>
            <Button type = "submit" onClick = { () => { setFieldValue("submitType", "stop-testing") } } style = { { width: "220px", backgroundColor: "#38512f", borderColor: "#38512f", color: "white", opacity: "90%" } }>
              <StopIcon style = { { fontSize: "14px" } }/>
              <span style = { { marginTop: "4px", marginLeft: "4px", marginRight: "12px", fontSize: "10px", textAlign: "center"} }>
                Stop Testing
              </span>
            </Button>
          </div>
        }

      </ButtonGroup>

    </Form>
  )
}

export default ModelTesting