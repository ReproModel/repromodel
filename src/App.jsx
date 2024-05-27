import { useState } from "react";
import { Box, Grid, Typography } from "@mui/material";
import { Formik } from "formik";
import { Button, ButtonGroup } from "@mui/material";

import "./App.css";

import Header from "./components/header/header"
import RepromodelStructure from "./components/repromodel-structure/repromodel-structure"
import newQuestions from "../repromodel_core/choices.json";
import { handleFileChange, handleSubmit } from "./utils/json-helpers";
import DynamicFormBuilder from "./UI/DynamicFormBuilder";
import ProgressViewer from "./UI/progress-viewer"
import TestingFormBuilder from "./UI/TestingFormBuilder";

function App() {
  // Generate initial values from questions data
  const initialValues = Object.values(newQuestions).reduce(
    (values, question) => {
      values[question.id] = ""; // Set initial value as empty
      return values;
    },
    {}
  );
  const [selectedSection, setSelectedSection] = useState("Experiment Builder");

  return (
    <>
      <Header />
      <Formik initialValues={initialValues} onSubmit={handleSubmit}>
        {(FormikProps) => (
          <Grid container direction={"row"}>
            <Grid item xs={4} className="stackContainer">
              <Grid item className="stackFrame">
                <RepromodelStructure FormikProps = { FormikProps } />
              </Grid>
            </Grid>
            <Grid item xs={8} className="questionairContainer">
              <Box
                sx={{ position: "absolute", top: 32, right: 48, zIndex: 100 }}
              >
                <ButtonGroup variant="contained">
                  <Button
                    onClick={() => setSelectedSection("Experiment Builder")}
                  >
                    Experiment Builder
                  </Button>
                  <Button onClick={() => setSelectedSection("Model Testing")}>
                    Model Testing
                  </Button>
                  <Button onClick={() => setSelectedSection("Progress Viewer")}>
                    Progress Viewer
                  </Button>
                </ButtonGroup>
              </Box>

              <Grid item className="questionairFrame">
                {selectedSection === "Experiment Builder" && (
                  <>
                    <Typography variant="h4">Experiment Builder</Typography>
                    <DynamicFormBuilder
                      FormikProps={FormikProps}
                      setFieldValue={FormikProps.setFieldValue}
                      handleFileChange={handleFileChange}
                      newQuestions={newQuestions}
                    />
                  </>
                )}
                {selectedSection === "Progress Viewer" && (
                  <>
                    <Typography variant="h4">Progress Viewer</Typography>
                    <ProgressViewer />
                  </>
                )}
                {selectedSection === "Model Testing" && (
                  <>
                    <Typography variant="h4">

                      Model Testing
                    </Typography>
                    <TestingFormBuilder
                      FormikProps={FormikProps}
                      setFieldValue={FormikProps.setFieldValue}
                      handleFileChange={handleFileChange}
                      newQuestions={newQuestions}
                    />
                  </>
                )}
              </Grid>
            </Grid>
          </Grid>
        )}
      </Formik>
    </>
  );
}

export default App;
