import React from "react";
import { Formik, Form, Field } from "formik";

import NewFlexibleFormField from "./FormComponents/NewFlexibleFormField";
import "./FormComponents/Form.css";

import { Typography } from "@mui/material";
import { capitalizeFirstLetter } from "../helperFunctions/OtherHelper";
import { SmartFolderField } from "./FormComponents/FieldTypes/FolderQuestion";
import CustomSelect from "./FormComponents/FieldTypes/CustomSelectComponent";

const options = [
  { value: "chocolate", label: "Chocolate" },
  { value: "strawberry", label: "Strawberry" },
  { value: "vanilla", label: "Vanilla" },
];

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

          {/* Check wether it is a nestedfolder or a flat one */}
          {nestedFolders.includes(folder) ? (
            <>
              <SmartFolderField folder={folder} folderContent={folderContent} />

              {/* For each Filename */}
              {Object.entries(folderContent).map(([file, fileContent]) => (
                <>
                  {/* For each Class in the File */}
                  {Object.entries(fileContent).map(
                    ([className, classContent]) => (
                      <div style={{ paddingLeft: "16px" }}>
                        {FormikProps.values[folder] &&
                          FormikProps.values[folder].includes(className) && (
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
          ) : folder === "data_splits" ? (
            <div className="paramBox">
              {Object.entries(folderContent).map(([Param, value]) => (
                <>
                  <label htmlFor={`${folder}:${Param}`}>{Param}:</label>
                  <NewFlexibleFormField
                    id={`${folder}:${Param}`}
                    object={value}
                    type={value.type}
                    name={`${folder}:${Param}`}
                    label={Param}
                  />
                </>
              ))}
            </div>
          ) : folder === "metrics" ? (

            <>

            {/** Currently doesn't do anything since the metric is catched above */}
              {Object.entries(folderContent).map(([file, fileContent]) => (
                <div
                  style={{ display: "flex", flexDirection: "column" }}
                  key={file}
                >
                  {file !== "torchmetricswrewe" ? (
                    <>
                      {Object.entries(fileContent).map(
                        ([className, fileDetails]) => (
                          <label key={className}>
                            {" "}
                            {/* Unique key for child list */}
                            <Field
                              type="checkbox"
                              name={folder}
                              value={className}
                            />
                            {className}
                          </label>
                        )
                      )}
                    </>
                  ) : (
                    <>
                      <label key="torchmetrics">
                        <Field
                          type="checkbox"
                          name={folder}
                          value="TorchMetrics"
                        />
                        TorchMetrics
                      </label>
                    </>
                  )}
                </div>
              ))}

              {/** Handle the normal options beside Torchmetrics */}
              {/* For each Filename */}
              {Object.entries(folderContent).map(([file, fileContent]) => (
                <>
                  {/* For each Class in the File */}
                  {Object.entries(fileContent).map(
                    ([className, classContent]) => (
                      <div style={{ paddingLeft: "16px" }}>
                        {FormikProps.values[folder] &&
                          FormikProps.values[folder].includes(className) && (
                            <div className="paramBox">
                              <p>{className} Params</p>

                              {/* Conditionally renders Param Questions if the class is selected or torchmetric Options */}

                              <>
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
                                )}{" "}
                              </>
                            </div>
                          )}
                      </div>
                    )
                  )}
                </>
              ))}

              {/** Handle the TorchMetric Options */}
              {FormikProps.values[folder] &&
                FormikProps.values[folder].includes("TorchMetrics") && (
                  <div style={{ paddingLeft: "16px" }}>
                    <CustomSelect
                      name="flavor"
                      options={options}
                      placeholder="Select a flavor"
                      isMulti={false}
                    />
                  </div>
                )}
            </>
          ) : (
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
          )}
        </div>
      ))}

      <button type="submit">Submit</button>
    </Form>
  );
};

export default DynamicFormBuilder;
