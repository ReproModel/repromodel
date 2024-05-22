import React from "react";
import { Formik, Form, Field } from "formik";

import NewFlexibleFormField from "./FormComponents/NewFlexibleFormField";
import "./FormComponents/Form.css";

import { Typography } from "@mui/material";
import { capitalizeFirstLetter } from "../helperFunctions/OtherHelper";
import { SmartFolderField } from "./FormComponents/FieldTypes/FolderQuestion";

const foldersPartOfTesting = ["models", "datasets", "metrics", "device"];

const nestedFolders = [
  "models",
  "datasets",
  "preprocessing",
  "postprocessing",
  "metrics",
  "losses",
  "augmentations",
  "lr_schedulers",
  "optimizers",
  "early_stopping",
];

const DynamicFormBuilder = ({
  FormikProps,
  handleFileChange,
  newQuestions,
  setFieldValue,
}) => {
  return (
    <Form>
      {/* Optional JSON File upload */}
      <Typography>
        Upload existing Training config File and reduce selections if desired
      </Typography>
      <input
        type="file"
        id="uploadedJson"
        accept=".json"
        onChange={(event) => handleFileChange(event, setFieldValue)}
      />
      {/* Hidden field to capture the submit type */}
      <Field type="hidden" name="submitType" />
      {/* For each Folder */}

      {Object.entries(newQuestions).map(([folder, folderContent]) => (
        <>
          {foldersPartOfTesting.includes(folder) && (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <h4> {capitalizeFirstLetter(folder)}</h4>

              {/* Check wether it is a nestedfolder or a flat one */}
              {foldersPartOfTesting.includes(folder) &&
              nestedFolders.includes(folder) ? (
                <>
                  <SmartFolderField
                    folder={folder}
                    folderContent={folderContent}
                  />

                  {/* For each Filename */}
                  {Object.entries(folderContent).map(([file, fileContent]) => (
                    <>
                      {/* For each Class in the File */}
                      {Object.entries(fileContent).map(
                        ([className, classContent]) => (
                          <div style={{ paddingLeft: "16px" }}>
                            {FormikProps.values[folder] && folder === "metrics" &&
                              ((Array.isArray(FormikProps.values[folder]) &&
                                FormikProps.values[folder].includes(
                                  className
                                )) ||
                                (typeof FormikProps.values[folder] ===
                                  "string" &&
                                  FormikProps.values[folder] ===
                                    className)) && (
                                <div className="paramBox">
                                  <p>{className} Params</p>

                                  {/* Conditionally renders Param Questions if the class is selected */}

                                  {Object.entries(classContent).map(
                                    ([Param, value]) => (
                                      <>
                                        <label
                                          htmlFor={`${folder}_params:${className}:${Param}`}
                                        >
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
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                        )
                      )}
                    </>
                  ))}
                </>
              ) : foldersPartOfTesting.includes(folder) ? (
                <>
                  <label htmlFor={`${folder}`}>{folder}:</label>
                  <NewFlexibleFormField
                    id={`${folder}`}
                    object={folderContent}
                    type={folderContent.type}
                    name={`${folder}`}
                    label={folder}
                  />
                </>
              ) : (
                <></>
              )}
            </div>
          )}
        </>
      ))}
      <div style={{ display: "flex", flexDirection: "column" }}>
        <h4> Model Checkpoints</h4>
        {FormikProps.values["models"] &&
          FormikProps.values["models"].map((model, index) => (
            <>
              <label htmlFor={`checkpoints:${model}`}>{model}:</label>
              <Field
                key={index}
                placeholder="Path to Checkpoint"
                name={`checkpoints:${model}`}
                label={model}
              />
            </>
          ))}
      </div>

      <button
        type="submit"
        onClick={() => setFieldValue("submitType", "testing")}
      >
        Submit for Testing
      </button>
    </Form>
  );
};

export default DynamicFormBuilder;
