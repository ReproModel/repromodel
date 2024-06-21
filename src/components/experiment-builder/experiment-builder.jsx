import "./experiment-builder.css"

import axios from "axios"
import FlexibleFormField from "../ui/flexible-form-field/flexible-form-field"
import StopIcon from '@mui/icons-material/Stop'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import React from "react"

import { Button, ButtonGroup, Typography } from "@mui/material"
import { capitalizeAndRemoveUnderscore } from "../../utils/string-helpers"
import { Field, Form } from "formik"
import { FolderField } from "../ui/folder-field"

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

const ExperimentBuilder = ({
  FormikProps,
  handleFileChange,
  newQuestions,
  setFieldValue
}) => {

  const [trainingInProgress, setTrainingInProgress] = React.useState(false)

  React.useEffect(() => {
    
    const interval = setInterval(() => {
      
      axios.get("http://127.0.0.1:5005/ping")
        
        .then(response => {
          if (response.data.trainingInProgress === true) {
            setTrainingInProgress(true)
          } else {
            setTrainingInProgress(false)
          }
        })      

        .catch(error => {
          setTrainingInProgress(false)
        })

    }, 3000)

    return () => clearInterval(interval)

  }, [])

  return (
    <Form> 
      
      {/* Optional JSON file upload input. */}
      <Typography className="json-input-file-label">
        Optionally upload existing configuration file.
      </Typography>

      <input
        type="file"
        id="uploadedJson"
        className="json-input-file"
        accept=".json"
        onChange={(event) => handleFileChange(event, setFieldValue)}
      />

      {/* Hidden field used to capture the submitType. */}
      <Field type="hidden" name="submitType" />

      {/* Loop each folder. */}
      {Object.entries(newQuestions).map(([folder, folderContent]) => (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {folder !== "tags" && <h4> {capitalizeAndRemoveUnderscore(folder)}</h4>} 

          {/* Case 1: Folder is nested. */}
          {nestedFolders.includes(folder) ? (
            <>
              <FolderField folder={folder} folderContent={folderContent} />

              {/* Loop each file in each folder. */}
              {Object.entries(folderContent).map(([file, fileContent]) => (
                <>
                  {/* Loop each class in each file. */}
                  {Object.entries(fileContent).map(
                    ([className, classContent]) => (
                      <div style={{ paddingLeft: "16px" }}>
                        {FormikProps.values[folder] &&
                          ((Array.isArray(FormikProps.values[folder]) &&
                            FormikProps.values[folder].includes(
                              `${file}>${className}`
                            )) ||
                            (typeof FormikProps.values[folder] === "string" &&
                              FormikProps.values[folder] ===
                                `${file}>${className}`)) && (
                            <div className="param-box">
                              <p>{className} Params</p>

                              {/* Conditionally render param questions if the class is selected. */}
                              {Object.entries(classContent).map(
                                ([param, value]) => (
                                  <>
                                    <label
                                      htmlFor={`${folder}_params:${file}>${className}:${param}`}
                                    >
                                      {param}:
                                    </label>

                                    <FlexibleFormField
                                      id={`${folder}_params:${file}>${className}:${param}`}
                                      object={value}
                                      type={value.type}
                                      name={`${folder}_params:${file}>${className}:${param}`}
                                      label={param}
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
          ) : // Case 2: Folder is flat.
          folder === "data_splits" ? (
            <div className="param-box">
              {Object.entries(folderContent).map(([param, value]) => (
                <>
                  <label htmlFor={`${folder}:${param}`}>{param}:</label>

                  <FlexibleFormField
                    id={`${folder}:${param}`}
                    object={value}
                    type={value.type}
                    name={`${folder}:${param}`}
                    label={param}
                  />
                </>
              ))}
            </div>
          ) : (
            <>
              {folder !== "tags" && ( // exclude the tags, since they are not supposed to be rednerde
                <>
                  <label htmlFor={`${folder}`}>{folder}:</label>

                  <FlexibleFormField
                    id={`${folder}`}
                    object={folderContent}
                    type={folderContent.type}
                    name={`${folder}`}
                    label={folder}
                  />
                </>
              )}
            </>
          )}
        </div>
      ))}

      <ButtonGroup variant = "outlined" sx = { { marginTop: "16px" } }>
        
        {/* Start Training Button */}
        { !trainingInProgress && 
          <Button
            type = "submit"
            style = { { width: "170px" } }
            onClick = { () => setFieldValue("submitType", "training") }
          >
            <PlayArrowIcon />
            Train
          </Button>  
        }

        {/* Stop Training Button */}
        { trainingInProgress && 
          <Button
            type = "submit"
            style = { { width: "170px" } }
            onClick = { () => setFieldValue("submitType", "stop-training") }
          >
            <StopIcon/>
            Stop Training
          </Button>
        }
        
        {/* Download Config Button */}
        <Button
          type = "submit"
          onClick = { () => { setFieldValue("submitType", "download")} }
        >
          Download Config File
        </Button>

      </ButtonGroup>
    </Form>
  )
}

export default ExperimentBuilder