import React from "react";
import { Formik, Form, Field } from "formik";

import NewFlexibleFormField from "./FormComponents/NewFlexibleFormField";
import "./FormComponents/Form.css";
import newQuestions from "../choicesJSON/newQuestionsFormat.json";
import { Typography } from "@mui/material";
import { capitalizeFirstLetter } from "../helperFunctions/OtherHelper";
import { SmartFolderField } from "./FormComponents/FieldTypes/FolderQuestion";


const DynamicFormBuilder = ({
  FormikProps,
  handleFileChange,
  newQuestions,
  setFieldValue
}) => {
  return (
    <Form>
      {/* Optional JSON File upload */}
      <Typography>Optionally upload existing config File</Typography>
      <input
        type="file"
        id="uploadedJson"
        accept=".json"
        onChange={(event) => handleFileChange(event, setFieldValue)}
      />
      {/* For each Folder */}
      {Object.entries(newQuestions).map(([folder, folderContent]) => (
        <div style={{ display: "flex", flexDirection: "column" }}>
          <h4> {capitalizeFirstLetter(folder)}</h4>

          <SmartFolderField folder={folder} folderContent={folderContent}/>

          
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
                            <label htmlFor={`${folder}_params:${className}:${Param}`}>
                              {Param}:
                            </label>
                            <NewFlexibleFormField
                              id={`${folder}_params:${className}:${Param}`}
                              object={value}
                              type={value.type}
                              name={`${folder}_params:${className}:${Param}`}
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
