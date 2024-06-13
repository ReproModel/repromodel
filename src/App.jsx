import "./App.css"

import CustomScript from "./components/custom-script/custom-script"
import ExperimentBuilder from "./components/experiment-builder/experiment-builder"
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
              
              <Box sx = { { position: "absolute", top: 32, right: 48, zIndex: 100 } }>
                <ButtonGroup variant = "contained">
                  <Button variant = "outlined" style = { { backgroundColor : "white", fontSize: "12px" } } onClick = { () => setSelectedSection("Custom Script") }>Custom Script</Button>
                  <Button onClick = { () => setSelectedSection("Experiment Builder") } style = { { fontSize: "12px", opacity: (selectedSection == "Experiment Builder") ? "100%" : "70%"} }>Experiment Builder</Button>
                  <Button onClick = { () => setSelectedSection("Model Testing") } style = { { fontSize: "12px", opacity: (selectedSection == "Model Testing") ? "100%" : "70%"} }>Model Testing</Button>
                  <Button onClick = { () => setSelectedSection("Progress Viewer") } style = { { fontSize: "12px", opacity: (selectedSection == "Progress Viewer") ? "100%" : "70%"} }>Progress Viewer</Button>
                  <Button onClick = { () => setSelectedSection("LLM Description") } style = { { fontSize: "12px", opacity: (selectedSection == "LLM Description") ? "100%" : "70%"} }>LLM Description</Button>
                </ButtonGroup>
              </Box>

              <Grid item className = "tabs">

                { selectedSection === "Custom Script" && (
                  <>
                    <Typography variant = "h4" style = { { marginLeft: "10px", marginTop: "20px" } }>Create Custom Script</Typography>
                    <CustomScript/>
                  </>
                )}

                { selectedSection === "Experiment Builder" && (
                  <>
                    <Typography variant = "h4" style = { { marginTop: "20px" } }>Experiment Builder</Typography>
                    <ExperimentBuilder
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                      newQuestions = { newQuestions }
                   />
                  </>
                )}

                { selectedSection === "Model Testing" && (
                  <>
                    <Typography variant = "h4" style = { { marginTop: "20px" } }>Model Testing</Typography>
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
                    <Typography variant = "h4" style = { { marginTop: "20px" } }>Progress Viewer</Typography>
                    <ProgressViewer/>
                  </>
                )}
                
                { selectedSection === "LLM Description" && (
                  <>
                    <Typography variant = "h4" style = { { marginTop: "20px" } }>LLM Description</Typography>
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