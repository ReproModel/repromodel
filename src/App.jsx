import { useState } from "react";
import "./App.css";
import { Box, Grid, Typography } from "@mui/material";
import Header from "./UI/Header";
import { Formik, Form, Field } from "formik";
import NewFlexibleFormField from "./UI/FormComponents/NewFlexibleFormField";
import { Button, ButtonGroup } from "@mui/material";

import ModuleStack from "./UI/ModuleStack";
import NewDynamicFormBuilder from "./UI/NewDynamicFormBuilder";
import newQuestions from "../repromodel_core/choices.json";
import { capitalizeFirstLetter } from "./helperFunctions/OtherHelper";
import { handleFileChange, handleSubmit } from "./helperFunctions/FormHelper";
import DynamicFormBuilder from "./UI/NewDynamicFormBuilder";
import TrainingViewer from "./UI/TrainingViewer";

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
                <ModuleStack FormikProps={FormikProps} />
              </Grid>
            </Grid>
            <Grid item xs={8} className="questionairContainer">
              <Box sx={{ position: "absolute", top: 32, right: 48 , zIndex: 100}}>
                <ButtonGroup variant="contained">
                  <Button
                    onClick={() => setSelectedSection("Experiment Builder")}
                  >
                    Experiment Builder
                  </Button>
                  <Button onClick={() => setSelectedSection("Model Testing")}>
                    Model Testing
                  </Button>
                  <Button onClick={() => setSelectedSection("Training Viewer")}>
                    Training Viewer
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
                {selectedSection === "Training Viewer" && (
                  <>
                    <Typography variant="h4">Training Viewer</Typography>
                    <TrainingViewer />
                  </>
                )}
                {selectedSection === "Model Testing" && (
                  <>
                    <Typography variant="h4">Here comes the Training Viewer</Typography>
                    
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
