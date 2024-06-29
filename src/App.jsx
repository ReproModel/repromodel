import "./App.css"

import CustomScript from "./components/custom-script/custom-script"
import ExperimentBuilderWrapper from "./components/experiment-builder/experiment-builder-wrapper"
import ExtractCode from "./components/extract-code/extract-code"
import Header from "./components/header/header"
import LLMDescription from "./components/llm-description/llm-description"
import ModelTesting from "./components/model-testing/model-testing"
import newQuestions from "../repromodel_core/choices.json"
import ProgressViewer from "./components/progress-viewer/progress-viewer"
import RepromodelStructure from "./components/repromodel-structure/repromodel-structure"

import { Button, ButtonGroup } from "@mui/material"
import { Box, Grid, Typography } from "@mui/material"
import { Formik } from "formik"
import { handleFileChange, handleSubmit } from "./utils/json-helpers"
import { useState } from "react"

function App() {

  const [selectedSection, setSelectedSection] = useState("Experiment Builder")
  
  // Generate initial values from questions data.
  const initialValues = Object.values(newQuestions).reduce(
    (values, question) => {
      // Set initial value as empty.
      values[question.id] = ""
      return values
    }
  )

  return (
    <>
      <Header/>

      <Formik initialValues = { initialValues } onSubmit = { handleSubmit }>
        
        { (FormikProps) => (
          <Grid container direction = { "row" }>
            
            <Grid item xs = { 4 } className = "repromodal-structure-container">
              <Grid item className = "repromodal-structure">
                <RepromodelStructure FormikProps = { FormikProps }/>
              </Grid>
            </Grid>
            
            <Grid item xs = { 8 } className = "tabs-container">
              

              <Box sx = { { position: "absolute", top: 16, right: 16, left: 0, zIndex: 100 } }>
                <ButtonGroup variant = "contained" sx = { { width: "100%", backgroundColor : "white" } }>
                  <Button variant = "outlined" style = { { backgroundColor: "white", color: "#38512f", borderColor: "#38512f", fontSize: "9px", width: "16.66%" } } onClick = { () => setSelectedSection("Custom Script") }>Custom Script</Button>
                  <Button onClick = { () => setSelectedSection("Experiment Builder") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%", opacity: (selectedSection == "Experiment Builder") ? "100%" : "70%"} }>Experiment Builder</Button>
                  <Button onClick = { () => setSelectedSection("Model Testing") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%", opacity: (selectedSection == "Model Testing") ? "100%" : "70%"} }>Model Testing</Button>
                  <Button onClick = { () => setSelectedSection("Progress Viewer") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%",opacity: (selectedSection == "Progress Viewer") ? "100%" : "70%"} }>Progress Viewer</Button>
                  <Button onClick = { () => setSelectedSection("Extract Code") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%",opacity: (selectedSection == "Extract Code") ? "100%" : "70%"} }>Extract Code</Button>
                  <Button onClick = { () => setSelectedSection("LLM Description") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%", opacity: (selectedSection == "LLM Description") ? "100%" : "70%"} }>LLM Description</Button>
                </ButtonGroup>
              </Box>

              <Grid item className = "tabs" style = { { marginTop : "4px" } }>

                { selectedSection === "Custom Script" && (
                  <>
                    <Typography variant = "h6" style = { { marginTop: "24px", fontWeight: "600" } }>Create Custom Script</Typography>
                    <CustomScript/>
                  </>
                )}

                { selectedSection === "Experiment Builder" && (
                  <>
                    <Typography variant = "h6" style = { { marginTop: "24px", fontWeight: "600" } }>Experiment Builder</Typography>
                    <ExperimentBuilderWrapper
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                      newQuestions = { newQuestions }
                   />
                  </>
                )}

                { selectedSection === "Model Testing" && (
                  <>
                    <Typography variant = "h6" style = { { marginTop: "24px", fontWeight: "600" } }>Model Testing</Typography>
                    <ModelTesting
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                      newQuestions = { newQuestions }
                   />
                  </>
                )}

                { selectedSection === "Progress Viewer" && (
                  <>
                    <Typography variant = "h6" style = { { marginTop: "36px", fontWeight: "600" } }>Progress Viewer</Typography>
                    <ProgressViewer/>
                  </>
                )}

                { selectedSection === "Extract Code" && (
                  <>
                    <Typography variant = "h6" style = { { marginTop: "36px", fontWeight: "600" } }>Extract Code</Typography>
                    <ExtractCode/>
                  </>
                )}
                
                { selectedSection === "LLM Description" && (
                  <>
                    <Typography variant = "h6" style = { { marginTop: "36px", fontWeight: "600" } }>LLM Description</Typography>
                    <LLMDescription
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                    />
                  </>
                )}

              </Grid>

            </Grid>

          </Grid>

        )}

      </Formik>
    </>
  )
}

export default App