import React from "react";
import { useState } from "react";

import { Button, ButtonGroup, Typography } from "@mui/material";
import { capitalizeFirstLetter } from "../../utils/string-helpers";
import { Field, Form, Formik } from "formik";
import { FolderField } from "../ui/folder-field";
import ExperimentBuilder from "./experiment-builder";
import ModalitySection from "../modality-section/modality-section";

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

const ExperimentBuilderWrapper = ({
  FormikProps,
  handleFileChange,
  newQuestions,
  setFieldValue,
}) => {
  const [filterChoosen, setFilterChoosen] = useState(false);
  return (
    <>
      {!filterChoosen && (
        <>
          <ModalitySection tags={newQuestions.tags}/>
          <button onClick={() => setFilterChoosen(true)}>Start Building</button>
        </>
      )}
      {filterChoosen && (
        <ExperimentBuilder
          FormikProps={FormikProps}
          setFieldValue={FormikProps.setFieldValue}
          handleFileChange={handleFileChange}
          newQuestions={newQuestions}
        />
      )}
    </>
  );
};

export default ExperimentBuilderWrapper;
