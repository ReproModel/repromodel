import "./modality-options.css";

import ModalityOptions from "./modality-options";
import React from "react";

import { capitalizeFirstLetter } from "../../utils/string-helpers";
import { Button, Typography } from "@mui/material";
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

    if (data[key] && data[key][values]) {
      if (data[key][values][typeOfWhatIWantToGet]) {
        models = data[key][values][typeOfWhatIWantToGet];
      } else {
       return null //return null when there is no model that fits that criterion e.g. for regression. 
      }
    }

    return models;
  }

  // Get models for each selected option.
  const taskModels = getModelsForKey("task", task);
  const subtaskModels = getModelsForKey("subtask", subtask);
  const modalityModels = getModelsForKey("modality", modality);
  const submodalityModels = getModelsForKey("submodality", submodality);

  if (taskModels === null || subtaskModels === null || modalityModels === null || submodalityModels === null) {
    return -1; //return null when there is no model that fits that criterion e.g. for regression. 
  }

  console.log("Task Models:", taskModels);
  console.log("Subtask Models:", subtaskModels);
  console.log("Modality Models:", modalityModels);
  console.log("Submodality Models:", submodalityModels);

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

  // Check if any of the models are greater than 0 but intersectedModels is 0
  if (
    (taskModels.length > 0 ||
      subtaskModels.length > 0 ||
      modalityModels.length > 0 ||
      submodalityModels.length > 0) &&
    intersectedModels.length === 0
  ) {
    return -1;
  }

  return intersectedModels;
}

///////////////////////////////////////////////////////////
// Get total number of models
///////////////////////////////////////////////////////////

function countUniqueModelsAndDatasets(data) {
  const modelsSet = new Set();
  const datasetsSet = new Set();

  function extractModelsAndDatasets(section) {
    if (typeof section === "object" && section !== null) {
      for (const key in section) {
        const value = section[key];
        if (value && typeof value === "object") {
          if (Array.isArray(value.models)) {
            value.models.forEach((model) => modelsSet.add(model));
          }
          if (Array.isArray(value.datasets)) {
            value.datasets.forEach((dataset) => datasetsSet.add(dataset));
          }
          extractModelsAndDatasets(value);
        }
      }
    }
  }

  extractModelsAndDatasets(data);

  return { uniqueModels: modelsSet.size, uniqueDatasets: datasetsSet.size };
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

  const totalAvailable = countUniqueModelsAndDatasets(class_per_tag);
  const modelsAvailable = totalAvailable.uniqueModels;
  const datasetsAvailable = totalAvailable.uniqueDatasets;

  const renderModelCount = () => {
    if (selectedModels.length === 0) {
      return modelsAvailable;
    } else if (selectedModels === -1) {
      return 0;
    } else {
      return selectedModels.length;
    }
  };

  return (
    <div>
      <p className="model-count" style={{ fontSize: "14px" }}>
        Available Models: {renderModelCount()}
      </p>

      <div className="container">
        {Object.entries(class_per_tag).map(([category, options], idx) => (
          <div style={{ marginTop: idx == 0 ? "0" : "24px" }}>
            <Typography
              style={{
                marginBottom: "12px",
                fontSize: "24px",
                marginLeft: "2px",
              }}
            >
              <span className="choose-modality-label">
                Choose <strong>{capitalizeFirstLetter(category)}</strong>
              </span>
            </Typography>

            <ModalityOptions
              options={options}
              onOptionClick={handleOptionClick}
              selectedOptions={selectedOptions[category] || []}
              group={category}
            />
          </div>
        ))}
      </div>
    {/**  <p>
        {Array.isArray(selectedModels) && selectedModels.length > 0 ? (
          selectedModels.map((str, index) => (
            <span key={index}>
              {str}
              <br />
            </span>
          ))
        ) : (
          <span>No models selected</span>
        )}
      </p>*/} 
      <div className="button-row" style={{ marginTop: "4px" }}>
        {selectedModels.length > 0 ? (
          <div>
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
          </div>
        ) : (
          <div>
            <button
              type="submit"
              className="skip-filter-button"
              onClick={() => {
                setFilterChoosen(true);
                setSelectedModels(selectedModels);
              }}
            >
              Skip Filter
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModalitySection;
