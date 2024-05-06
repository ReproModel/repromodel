import React from "react";
import { Formik, Form, Field } from "formik";

import NewFlexibleFormField from "./FormComponents/NewFlexibleFormField";
import "./FormComponents/Form.css";
import newQuestions from "../choicesJSON/newQuestionsFormat.json";
import { Typography } from "@mui/material";
import { capitalizeFirstLetter } from "../helperFunctions/OtherHelper";

const DynamicFormBuilder = ({
  FormikProps,
  handleFileChange,
  newQuestions,
}) => {
  return (
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
              {Object.entries(fileContent).map(([className, fileContent]) => (
                <label>
                  <Field
                    type="checkbox"
                    name={folder}
                    value={className}
                  ></Field>
                  {className}
                </label>
              ))}
            </>
          ))}

          {/* For each Filename */}
          {Object.entries(folderContent).map(([file, fileContent]) => (
            <>
              {/* For each Class in the File */}
              {Object.entries(fileContent).map(([className, classContent]) => (
                <div style={{ paddingLeft: "16px" }}>
                  {FormikProps.values[folder] &&
                    FormikProps.values[folder].includes(className) && (
                      <div className="paramBox">
                        <p>{className} Params</p>

                        {/* Conditionally renders Param Questions if the class is selected */}

                        {Object.entries(classContent).map(([Param, value]) => (
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
                        ))}
                      </div>
                    )}
                </div>
              ))}
            </>
          ))}
        </div>
      ))}

      <button type="submit">Submit</button>
    </Form>
  );
};

export default DynamicFormBuilder;
