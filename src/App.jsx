import "./App.css"

import ExperimentBuilder from "./components/experiment-builder/experiment-builder"
import Header from "./components/header/header"
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
                  <Button onClick={() => setSelectedSection("Experiment Builder")}>Experiment Builder</Button>
                  <Button onClick={() => setSelectedSection("Model Testing")}>Model Testing</Button>
                  <Button onClick={() => setSelectedSection("Progress Viewer")}>Progress Viewer</Button>
                </ButtonGroup>
              </Box>

              <Grid item className = "tabs">
                
                { selectedSection === "Experiment Builder" && (
                  <>
                    <Typography variant = "h4">Experiment Builder</Typography>
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
                    <Typography variant = "h4">Model Testing</Typography>
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
                    <Typography variant = "h4">Progress Viewer</Typography>
                    <ProgressViewer/>
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