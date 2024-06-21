import CustomSelect from "./custom-select";
import React from "react";

const extractSecondLevelKeys = (obj) => {
  const options = [];

  const traverse = (obj, level = 0) => {
    if (level === 1) {
      Object.keys(obj).forEach((key) => {
        options.push({ value: key, label: key });
      });
    } else {
      Object.values(obj).forEach((value) => {
        if (typeof value === "object") {
          traverse(value, level + 1);
        }
      });
    }
  };

  traverse(obj);

  return options;
};

const extractFirstandSecondLevelKeys = (obj, noneOptionFolder) => {
  const options = [];

  if (noneOptionFolder) {
    options.push({ value: "", label: "--- None ---" });
  }

  const traverse = (obj, level = 0, parentKey = "") => {
    if (level === 1) {
      Object.keys(obj).forEach((key) => {
        options.push({ value: `${parentKey}>${key}`, label: `${parentKey}>${key}` })
      });
    } else {
      Object.entries(obj).forEach(([key, value]) => {
        if (typeof value === "object") {
          traverse(value, level + 1, level === 0 ? key : parentKey);
        }
      });
    }
  };

  traverse(obj);

  return options;
};
export function FolderField({ folder, folderContent }) {
  // Define the array of specific folder names.
  const multipleSelectFolders = ["models", "metrics"];

  // Define the Folders that should be able to have "None" as an option
  const foldersWithNoneOption = ["preprocessing", "postprocessing"];
  const noneOptionFolder = foldersWithNoneOption.includes(folder);

  // Convert folderContent into array.
  const newoptions = extractFirstandSecondLevelKeys(
    folderContent,
    noneOptionFolder
  );

  // Check if the folder name is within the specific folders array.
  const isMultipleFolder = multipleSelectFolders.includes(folder);

  return (
    <CustomSelect
      name={folder}
      options={newoptions}
      placeholder="Select or start typing..."
      isMulti={isMultipleFolder}
    />
  );
}
