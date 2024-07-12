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
import MobileWarning from "./components/mobile/mobile-warning";

import { Button, ButtonGroup } from "@mui/material"
import { Box, Grid, Typography } from "@mui/material"
import { Formik } from "formik"
import { handleFileChange, handleSubmit } from "./utils/json-helpers"
import { useState, useEffect } from "react"

function App() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkIfMobile = () => {
      const userAgent =
        typeof window.navigator === "undefined" ? "" : navigator.userAgent;
      const mobile = /iPhone|iPod|Android/i.test(userAgent);
      setIsMobile(mobile);
    };

    checkIfMobile();
    window.addEventListener("resize", checkIfMobile);

    return () => {
      window.removeEventListener("resize", checkIfMobile);
    };
  }, []);

  const [selectedSection, setSelectedSection] = useState("Experiment Builder")
  
  // Generate initial values from questions data.
  const initialValues = Object.values(newQuestions).reduce((values, question) => {
    // Set initial value as empty for other questions.
    values[question.id] = "";
    return values;
  }, { testType: 'testing-crossValidation' }); // Set initial value for testType

  
  return (
    <>
      {isMobile ? (
        <MobileWarning />
      ) : (
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
              
              <Box sx = { { position: "absolute", top: 16, right: 16, left: 0, zIndex: 500 } }>
                <ButtonGroup variant = "contained" sx = { { width: "100%", backgroundColor : "white" } }>
                  <Button variant = "outlined" style = { { backgroundColor: "white", color: "#38512f", borderColor: "#38512f", fontSize: "9px", width: "16.66%" } } onClick = { () => setSelectedSection("Custom Script") }>Custom Script</Button>
                  <Button onClick = { () => setSelectedSection("Experiment Builder") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%", opacity: (selectedSection == "Experiment Builder") ? "100%" : "70%"} }>Experiment Builder</Button>
                  <Button onClick = { () => setSelectedSection("Model Testing") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%", opacity: (selectedSection == "Model Testing") ? "100%" : "70%"} }>Model Testing</Button>
                  <Button onClick = { () => setSelectedSection("Progress Viewer") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%",opacity: (selectedSection == "Progress Viewer") ? "100%" : "70%"} }>Progress Viewer</Button>
                  <Button onClick = { () => setSelectedSection("Extract Code") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%",opacity: (selectedSection == "Extract Code") ? "100%" : "70%"} }>Extract Code</Button>
                  <Button onClick = { () => setSelectedSection("LLM Description") } style = { { backgroundColor: "#38512f", borderColor: "#162012", borderStyle: "dotted", fontSize: "9px", width: "16.66%", opacity: (selectedSection == "LLM Description") ? "100%" : "70%"} }>LLM Description</Button>
                </ButtonGroup>
              </Box>

              <Grid item className = "tabs">

                { selectedSection === "Custom Script" && (
                  <div className = "tab-div">
                    <Typography variant = "h6">
                      <span className = "tab-header">Create Custom Script</span>
                    </Typography>
                    <CustomScript/>
                  </div>
                )}

                { selectedSection === "Experiment Builder" && (
                  <div className = "tab-div">
                    <Typography variant = "h6">
                      <span className = "tab-header">Experiment Builder</span>
                    </Typography>
                    <ExperimentBuilderWrapper
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                      newQuestions = { newQuestions }
                   />
                  </div>
                )}

                { selectedSection === "Model Testing" && (
                  <div className = "tab-div">
                    <Typography variant = "h6">
                      <span className = "tab-header">Model Testing</span>
                    </Typography>
                    <ModelTesting
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                      newQuestions = { newQuestions }
                   />
                  </div>
                )}

                { selectedSection === "Progress Viewer" && (
                  <div className = "tab-div">
                    <Typography variant = "h6">
                      <span className = "tab-header">Progress Viewer</span>
                    </Typography>
                    <ProgressViewer/>
                  </div>
                )}

                { selectedSection === "Extract Code" && (
                  <div className = "tab-div">
                    <Typography variant = "h6">
                      <span className = "tab-header">Extract Code</span>
                    </Typography>
                    <ExtractCode/>
                  </div>
                )}
                
                { selectedSection === "LLM Description" && (
                  <div className = "tab-div">
                    <Typography variant = "h6">
                      <span className = "tab-header">LLM Description</span>
                    </Typography>
                    <LLMDescription
                      FormikProps = { FormikProps }
                      setFieldValue = { FormikProps.setFieldValue }
                      handleFileChange = { handleFileChange }
                    />
                  </div>
                )}

              </Grid>

            </Grid>

          </Grid>

        )}

      </Formik>
      </>
      )}
    </>
  )
}

export default App