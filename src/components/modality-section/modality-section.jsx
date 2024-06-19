import "./modality-options.css";

import ModalityOptions from "./modality-options";
import React from "react";

import { capitalizeFirstLetter } from "../../utils/string-helpers";
import { Typography } from "@mui/material";
import { useState } from "react";

///////////////////////////////////////////////////////////
// getModels()
///////////////////////////////////////////////////////////

function getModels(selectedOptions, jsonData) {
  const { task, subtask, modality, submodality } = selectedOptions;
  const data = jsonData;

  // Helper function to get models for a given key and values.
  function getModelsForKey(key, values) {
    // To be used dynamically in the future.
    const typeOfWhatIWantToGet = "models";

    let models = [];

    if (
      data[key] &&
      data[key][values] &&
      data[key][values][typeOfWhatIWantToGet]
    ) {
      models = data[key][values][typeOfWhatIWantToGet];
    }

    return models;
  }

  // Get models for each selected option.
  const taskModels = getModelsForKey("task", task);
  const subtaskModels = getModelsForKey("subtask", subtask);
  const modalityModels = getModelsForKey("modality", modality);
  const submodalityModels = getModelsForKey("submodality", submodality);

  /*  console.log("Task Models:", taskModels);
  console.log("Subtask Models:", subtaskModels);
  console.log("Modality Models:", modalityModels);
  console.log("Submodality Models:", submodalityModels);**/

  // Intersect all model arrays.
  function intersect(arr1, arr2) {
    if (arr1.length === 0) return arr2;
    if (arr2.length === 0) return arr1;
    return arr1.filter((value) => arr2.includes(value));
  }

  let intersectedModels = taskModels;

  if (subtaskModels.length > 0) {
    intersectedModels = intersect(intersectedModels, subtaskModels);
  }
  if (modalityModels.length > 0) {
    intersectedModels = intersect(intersectedModels, modalityModels);
  }
  if (submodalityModels.length > 0) {
    intersectedModels = intersect(intersectedModels, submodalityModels);
  }

  return intersectedModels;
}

///////////////////////////////////////////////////////////
// ModalitySection component
///////////////////////////////////////////////////////////

const ModalitySection = ({
  class_per_tag,
  setFilterChoosen,
  setSelectedModels,
}) => {
  const [selectedOptions, setSelectedOptions] = useState({
    task: [],
    subtask: [],
    modality: [],
    submodality: [],
  });

  const handleOptionClick = (group, option) => {
    setSelectedOptions((prevSelectedOptions) => {
      const groupOptions = prevSelectedOptions[group];

      // Check if the option is already selected for the group.
      if (groupOptions && groupOptions.includes(option)) {
        // Deselect the option if it is already selected.
        return { ...prevSelectedOptions, [group]: [] };
      } else {
        // Select the new option, replacing any previously selected option.
        return { ...prevSelectedOptions, [group]: [option] };
      }
    });
  };

  const selectedModels = getModels(selectedOptions, class_per_tag);

  return (
    <div className="container">
      <div className="button-row">
        {selectedModels.length > 0 ? (
          <>
            <button
              type="submit"
              className="start-building-button"
              onClick={() => {
                setFilterChoosen(true);
                setSelectedModels(selectedModels);
              }}
            >
              Start Building
            </button>
          </>
        ) : (
          <button
            type="submit"
            className="start-building-button"
            onClick={() => {
              setFilterChoosen(true);
              setSelectedModels(selectedModels);
            }}
          >
            Skip Filter
          </button>
        )}
        <p className="model-count">Filtered Models: {selectedModels.length}</p>
      </div>
      {Object.entries(class_per_tag).map(([category, options]) => (
        <>
          <Typography
            style={{ marginTop: "16px", marginBottom: "4px" }}
            variant="h7"
          >
            Choose {capitalizeFirstLetter(category)}
          </Typography>

          <ModalityOptions
            options={options}
            onOptionClick={handleOptionClick}
            selectedOptions={selectedOptions[category] || []}
            group={category}
          />
        </>
      ))}
    </div>
  );
};

export default ModalitySection;
