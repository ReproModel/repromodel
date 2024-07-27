import "./model-testing.css";

import axios from "axios";
import FlexibleFormField from "../ui/flexible-form-field/flexible-form-field";
import StopIcon from "@mui/icons-material/Stop";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import React from "react";

import { Button, ButtonGroup, FormControl, FormControlLabel, FormLabel, Radio, RadioGroup } from "@mui/material";
import { capitalizeFirstLetter } from "../../utils/string-helpers";
import { Field, Form } from "formik";
import { FolderField } from "../ui/folder-field";

const foldersPartOfTesting = [
  "models",
  "datasets",
  "metrics",
  "augmentations",
  "device",
  "batch_size",
  "tensorboard_log_path",
  "model_save_path",
];

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
];

const ModelTesting = ({
  FormikProps,
  handleFileChange,
  newQuestions,
  setFieldValue,
}) => {
  
  const [testingInProgress, setTestingInProgress] = React.useState(false);

  // Dark Theme
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  const [isDarkTheme, setIsDarkTheme] = React.useState(getCurrentTheme())

  const mqListener = (e => { setIsDarkTheme(e.matches) })

  React.useEffect(() => {
    const darkThemeMq = window.matchMedia("(prefers-color-scheme: dark)");
    darkThemeMq.addListener(mqListener)
    return () => darkThemeMq.removeListener(mqListener)
  }, [])

  React.useEffect(() => {
    const interval = setInterval(() => {
      axios
        .get("http://127.0.0.1:5005/ping")

        .then((response) => {
          if (response.data.cvTestingInProgress === true || response.data.finalTestingInProgress === true ) {
            setTestingInProgress(true);
          } else {
            setTestingInProgress(false);
          }
        })

        .catch((error) => {
          setTestingInProgress(false);
        });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Form>
      <p className = "test-type-label">
        Please choose between Cross-Validation Testing and Final Testing on
        unseen data.
      </p>

      <FormControl component="fieldset">
        <FormLabel component="legend" sx = { { paddingTop: "10px", color: isDarkTheme ? "white" : "black" } }>Test Type</FormLabel>
        <Field
          as={RadioGroup}
          aria-label="testType"
          name="testType"
          value={FormikProps.values.testType}
          onChange={(event) => {
            setFieldValue("testType", event.target.value);
          }}
        >
          <FormControlLabel
            value="testing-crossValidation"
            control={<Radio sx = { { color: (isDarkTheme ? "white !important" : "black"), "&.Mui-checked": { color: isDarkTheme ? "white !important" : "black" } } } style = { { opacity: "50%", color: "black", fontSize: "12px", root: { "& .MuiSvgIcon-root": { height: "1.5em", width: "2.5em" } } } }/>}
            label="Cross-Validation Testing"
          />
          <FormControlLabel
            value="testing-final"
            control={<Radio sx = { { color: (isDarkTheme ? "white !important" : "black"), "&.Mui-checked": { color: isDarkTheme ? "white !important" : "black" } } } style = { { opacity: "50%", color: "black", fontSize: "12px", root: { "& .MuiSvgIcon-root": { height: "1.5em", width: "2.5em" } } } } />}
            label="Final Testing"
          />
          
        </Field>
      </FormControl>

      {FormikProps.values.testType === "testing-final" && (
        <p
          style={{
            backgroundColor: "#f0f8ff",
            color: "#31708f",
            padding: "10px",
            borderRadius: "5px",
            border: "1px solid #bce8f1",
            fontSize: "14px",
            fontStyle: "italic",
          }}
        >
          Please be aware that you may only select a single model for final
          testing.
        </p>
      )}
      {/* Optional JSON file upload input. */}
      <div>
        <div className="json-input-file-label">
          <span
            style={{
              fontFamily:
                "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif",
              fontSize: "12px",
              fontWeight: "700",
            }}
          >
            Upload Existing Testing Configuration
          </span>
        </div>

        <input
          type="file"
          id="uploadedJson"
          className="json-input-file"
          accept=".json"
          onChange={(event) => handleFileChange(event, setFieldValue)}
        />
      </div>

      {/* Hidden field used to capture the submitType. */}
      <Field type="hidden" name="submitType" />

      {/* Loop each folder. */}
      {Object.entries(newQuestions).map(([folder, folderContent]) => (
        <div style={{ width: "50%" }}>
          {foldersPartOfTesting.includes(folder) && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                fontSize: "12px",
              }}
            >
              <h4 className="model-testing-folder-label">
                {" "}
                {capitalizeFirstLetter(folder)}
              </h4>

              {/* Case 1: Folder is nested and part of testing. */}
              {foldersPartOfTesting.includes(folder) &&
              nestedFolders.includes(folder) ? (
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
                              (folder === "models" || folder === "datasets" || folder === "augmentations")  &&
                              ((Array.isArray(FormikProps.values[folder]) &&
                                FormikProps.values[folder].includes(
                                   `${file}>${className}`
                                )) ||
                                (typeof FormikProps.values[folder] ===
                                  "string" &&
                                  FormikProps.values[folder] ===
                                  `${file}>${className}`)) && (
                                <div className="param-box">
                                  <p>{className} Params</p>

                                  {/* Conditionally render param questions if the class is selected. */}
                                  {Object.entries(classContent).map(
                                    ([param, value]) => (
                                      <>
                                        <label
                                         className = "param-label"
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
              ) : // Case 2: Folder is flat and part of testing.
              foldersPartOfTesting.includes(folder) ? (
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
              ) : (
                <></>
              )}
            </div>
          )}
        </div>
      ))}

      <ButtonGroup variant="outlined" sx={{ marginTop: "36px" }}>
        
        {/* Start Testing Button */}
        {!testingInProgress && (
          <div>
            <Button
              type="submit"
              onClick={() => {
                setFieldValue("submitType", "testing");
              }}
              style={{
                width: "220px",
                backgroundColor: "#38512f",
                borderColor: "#38512f",
                color: "white",
                opacity: "90%",
              }}
            >
              <PlayArrowIcon style={{ fontSize: "14px" }} />
              <span
                style={{
                  marginTop: "4px",
                  marginLeft: "12px",
                  marginRight: "12px",
                  fontSize: "12px",
                  textAlign: "center",
                }}
              >
                Test
              </span>
            </Button>
          </div>
        )}

        {/* Stop Testing Button */}
        {testingInProgress && (
          <div>
            <Button
              type="submit"
              onClick={() => {
                setFieldValue("submitType", "stop-testing");
              }}
              style={{
                width: "220px",
                backgroundColor: "#38512f",
                borderColor: "#38512f",
                color: "white",
                opacity: "90%",
              }}
            >
              <StopIcon style={{ fontSize: "14px" }} />
              <span
                style={{
                  marginTop: "4px",
                  marginLeft: "4px",
                  marginRight: "12px",
                  fontSize: "10px",
                  textAlign: "center",
                }}
              >
                Stop Testing
              </span>
            </Button>
          </div>
        )}
      </ButtonGroup>
    </Form>
  );
};

export default ModelTesting