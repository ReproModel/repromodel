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
        {(FormikProps ) => (
      <Grid container direction={"row"}>
        <Grid item xs={4} className="stackContainer">
          <Grid item className="stackFrame">
         
            <ModuleStack FormikProps={FormikProps}/>
          </Grid>
        </Grid>
        <Grid item xs={8} className="questionairContainer">
          <Grid item className="questionairFrame">
          <div>
      <h1>Experiment Builder</h1>
     
          <Form>
            {/* Optional JSON File upload */}
            <Typography>Optionally upload existing config File</Typography>
            <input
              type="file"
              id="uploadedJson"
              accept=".json"
              onChange={handleFileChange}
            />
            {/* For each Folder */}
            {Object.entries(newQuestions).map(([folder, folderContent]) => (
              <div style={{ display: "flex", flexDirection: "column" }}>
                <h4> {capitalizeFirstLetter(folder)}</h4>

                {Object.entries(folderContent).map(([file, fileContent]) => (
                  <>
                    {" "}
                    {Object.entries(fileContent).map(
                      ([className, fileContent]) => (
                        <label>
                          <Field
                            type="checkbox"
                            name={folder}
                            value={className}
                          ></Field>
                          {className}
                        </label>
                      )
                    )}
                  </>
                ))}

                {/* For each Filename */}
                {Object.entries(folderContent).map(([file, fileContent]) => (
                  <>
                    {/* For each Class in the File */}
                    {Object.entries(fileContent).map(
                      ([className, classContent]) => (
                        <div style={{ paddingLeft: "16px" }}>
                          {FormikProps.values[folder] &&
                            FormikProps.values[folder].includes(className) && (
                              <div
                                style={{
                                  display: "flex",
                                  flexDirection: "column",
                                  border: "2px solid #000000",
                                  borderRadius: "10px",
                                  padding: "8px",
                                  margin: "8px",
                                }}
                              >
                                <p>{className} Params</p>

                                {/* Conditionally renders Param Questions if the class is selected */}

                                {Object.entries(classContent).map(
                                  ([Param, value]) => (
                                    <>
                                      <label htmlFor={`${className}:${Param}`}>
                                        {Param}:
                                      </label>
                                      <NewFlexibleFormField
                                        id={`${className}:${Param}`}
                                        object={value}
                                        type={value.type}
                                        name={`${className}:${Param}`}
                                        label={Param}
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
              </div>
            ))}

            <button type="submit">Submit</button>
          </Form>
        
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
