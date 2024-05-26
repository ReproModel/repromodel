import React from "react";
import { Formik, Form, Field, useFormikContext } from "formik";

import questionsData from "../choicesJSON/questionsList.json"
import FlexibleFormField from "../UI/FormComponents/FlexibleFormField";
import "./FormComponents/Field.css";

const DynamicFormBuilder = () => {
  // Generate initial values from questions data
  const initialValues = questionsData.reduce((values, question) => {
    if (question.type === "dropdown") {
      values[question.id] = ""; // Set initial value as empty for dropdowns
    } else {
      values[question.id] = ""; // Default empty string for other types
    }
    return values;
  }, {});

  return (
    <div>
      <h1>Experiment Builder</h1>
      <Formik
        initialValues={initialValues}
        onSubmit={(values) => {
          console.log(values);
        }}
      >
        {({ values }) => (
          <Form>
            {questionsData.map((question, index) => (
              <div
                style={{ display: "flex", flexDirection: "column" }}
                key={question.id}
              >
                <label htmlFor={question.id}>{question.label}:</label>
                {question.type === "dropdown" ? (
                  <React.Fragment>
                    <Field
                      key={question.id}
                      as="select"
                      id={question.id}
                      name={question.id}
                      className="inputField"
                    >
                      <option value="">Select an option</option>
                      {question.options.map((option) => (
                        <React.Fragment key={option.name}>
                          <option value={option.name}>{option.name}</option>
                        </React.Fragment>
                      ))}
                    </Field>

                    {question.options.map((option) => (
                      <div style={{ paddingLeft: '16px' }}>
                        {values[question.id] === option.name &&
                          typeof option.moreQuestions !== "undefined" && (
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                              }}
                            >
                              {/* Conditionally Renders the param Questions if the question has "moreQuestions", the typeof prevents it to run if "moreQuestions is undefined" */}
                              <h4>{option.name} Params</h4>

                              {option.moreQuestions.map((moreQuestions) => (
                                <React.Fragment>
                                  <label htmlFor={moreQuestions.id}>
                                    {moreQuestions.name}:
                                  </label>
                                  <Field
                                  className="inputField"
                                    type={moreQuestions.type}
                                    id={moreQuestions.id}
                                    name={moreQuestions.id}
                                  />
                                </React.Fragment>
                              ))}
                            </div>
                          )}
                      </div>
                    ))}
                  </React.Fragment>
                ) : (
                  <FlexibleFormField
                    id={question.id}
                    label={question.label}
                    type={question.type}
                  />
                )}
              </div>
            ))}
            <button type="submit">Submit</button>
          </Form>
        )}
      </Formik>
    </div>
  );
};

export default DynamicFormBuilder;
