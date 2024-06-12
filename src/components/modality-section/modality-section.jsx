import "./modality-options.css";

import ModalityOptions from "./modality-options";
import React, { useState } from "react";

import { Typography } from "@mui/material";
import { capitalizeFirstLetter } from "../../utils/string-helpers";

const modalities = [
  {
    label: "Image",
    image:
      "https://production-media.paperswithcode.com/thumbnails/task/task-0000000509-66402dc1_C47uozM.jpg",
    numPapers: "2157 Datasets",
    href: "",
  },
  {
    label: "Video",
    image:
      "https://production-media.paperswithcode.com/thumbnails/task/6f9c5c9e-b5fc-4ce3-b423-a3b196f0252c.jpg",
    numPapers: "2157 Datasets",
    href: "",
  },
  {
    label: "Audio",
    image:
      "https://production-media.paperswithcode.com/thumbnails/method/method-0000000312-bb51f64c.jpg",
    numPapers: "2157 Datasets",
    href: "",
  },
  {
    label: "3D",
    image:
      "https://production-media.paperswithcode.com/icons/task/48d55b59-3af2-4a6d-a195-572f1d4a1867.jpg",
    numPapers: "2157 Datasets",
    href: "",
  },
  {
    label: "Text",
    image:
      "https://production-media.paperswithcode.com/icons/task/f3b5b381-ae14-4572-a44b-79c4aea62230.jpg",
    numPapers: "2157 Datasets",
    href: "",
  },
];

const tasks = [
  {
    label: "Semantic Segmentation",
    image:
      "https://production-media.paperswithcode.com/icons/task/b45b7a24-e2dd-47e2-9d1f-0f372e5d9074.jpg",
    numPapers: "180 models",
    href: "",
  },
  {
    label: "Classification",
    image:
      "https://production-media.paperswithcode.com/icons/task/0aa45ecb-2bb1-4c8d-bd0c-16b4d9de739d.jpg",
    numPapers: "180 models",
    href: "",
  },
  {
    label: "Object Detecttion",
    image:
      "https://production-media.paperswithcode.com/icons/task/dd004e56-bc49-4cc1-b0d5-186f2dd17ce8.jpg",
    numPapers: "180 models",
    href: "",
  },
];


function getModels(selectedOptions, jsonData) {
  const { task, subtask, modality, submodality } = selectedOptions;
  const data = jsonData;

  // Helper function to get models for a given key and values
  function getModelsForKey(key, values) {
    const typeOfWhatIWantToGet = "models" //to be used dynamically in the future
    let models = [];
    

    if (data.tags[key] && data.tags[key][values] && data.tags[key][values][typeOfWhatIWantToGet]) {
      models = data.tags[key][values][typeOfWhatIWantToGet];
    }

    return models;
  }

  // Get models for each selected option
  const taskModels = getModelsForKey("task", task);
  const subtaskModels = getModelsForKey("subtask", subtask);
  const modalityModels = getModelsForKey("modality", modality);
  const submodalityModels = getModelsForKey("submodality", submodality);

  /** 
  console.log("Task Models:", taskModels);
  console.log("Subtask Models:", subtaskModels);
  console.log("Modality Models:", modalityModels);
  console.log("Submodality Models:", submodalityModels);*/

  // Intersect all model arrays
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
function filterModels(jsonData, intersectedModels) {
  if (intersectedModels.length === 0) {
      return jsonData;
  }

  const newModels = {};

  Object.entries(jsonData.models).forEach(([modelKey, modelValue]) => {
      Object.entries(modelValue).forEach(([innerModelKey, innerModelValue]) => {
          const fullKey = `${modelKey}>${innerModelKey}`;
          if (intersectedModels.includes(fullKey)) {
              if (!newModels[modelKey]) {
                  newModels[modelKey] = {};
              }
              newModels[modelKey][innerModelKey] = innerModelValue;
          }
      });
  });

  return { ...jsonData, models: newModels };
}


const ModalitySection = (class_per_tag) => {
  const [selectedOptions, setSelectedOptions] = useState({
    task: [],
    subtask: [],
    modality: [],
    submodality: [],
  });

  const handleOptionClick = (group, option) => {
    setSelectedOptions((prevSelectedOptions) => {
      const groupOptions = prevSelectedOptions[group];

      // Check if the option is already selected for the group
      if (groupOptions && groupOptions.includes(option)) {
        // Deselect the option if it is already selected
        return { ...prevSelectedOptions, [group]: [] };
      } else {
        // Select the new option, replacing any previously selected option
        return { ...prevSelectedOptions, [group]: [option] };
      }
    });
  };

  const selectedModels = getModels(selectedOptions, class_per_tag);

  return (
    <div className="container">
      {Object.entries(class_per_tag).map(([tag, tagContent]) => (
        <>
          {Object.entries(tagContent).map(([category, options]) => (
            <>
              <Typography style={{ marginTop: "16px" }} variant="h6">
                Choose your {category}
              </Typography>
              <ModalityOptions
                options={options}
                onOptionClick={handleOptionClick}
                selectedOptions={selectedOptions[category] || []}
                group={category}
              />
            </>
          ))}
        </>
      ))}
      
      {console.log("The selected models are ", selectedModels)}
      <Typography style={{ marginTop: "16px" }} variant="h6">
        {" "}
        Choose your modality:
      </Typography>
      <ModalityOptions cardOptions={modalities} />

      <Typography style={{ marginTop: "64px" }} variant="h6">
        {" "}
        Choose your task:
      </Typography>
      <ModalityOptions cardOptions={tasks} />
    </div>
  );
};

export default ModalitySection;
