import { useState } from "react";
import "./App.css";
import { Box, Grid, Typography } from "@mui/material";
import Header from "./UI/Header";
import { Formik, Form, Field } from "formik";
import NewFlexibleFormField from "./UI/FormComponents/NewFlexibleFormField";

import ModuleStack from "./UI/ModuleStack";
import NewDynamicFormBuilder from "./UI/NewDynamicFormBuilder";
import newQuestions from "./choicesJSON/newQuestionsFormat.json";
import { capitalizeFirstLetter } from "./helperFunctions/OtherHelper";
import { handleFileChange, handleSubmit } from "./helperFunctions/FormHelper";
import DynamicFormBuilder from "./UI/NewDynamicFormBuilder";

function App() {
  // Generate initial values from questions data
  const initialValues = Object.values(newQuestions).reduce(
    (values, question) => {
      values[question.id] = ""; // Set initial value as empty
      return values;
    },
    {}
  );

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
              <Grid item className="questionairFrame">
                <div>
                  <h1>Experiment Builder</h1>
                  <DynamicFormBuilder
                    FormikProps={FormikProps}
                    setFieldValue={FormikProps.setFieldValue}
                    handleFileChange={handleFileChange}
                    newQuestions={newQuestions}
                  />
                </div>
              </Grid>
            </Grid>
          </Grid>
        )}
      </Formik>
    </>
  );
}

export default App;
