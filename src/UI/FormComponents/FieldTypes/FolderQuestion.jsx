import CustomSelect from "./CustomSelectComponent"
import React from "react"

const extractSecondLevelKeys = (obj) => {
  
  const options = []

  const traverse = (obj, level = 0) => {
    
    if (level === 1) {
      Object.keys(obj).forEach((key) => {
        options.push({ value: key, label: key })
      })
    
    } else {
      Object.values(obj).forEach((value) => {
        if (typeof value === "object") {
          traverse(value, level + 1)
        }
      })
    }
  }

  traverse(obj)

  return options
}

export function SmartFolderField({ folder, folderContent }) {
  
  // Define the array of specific folder names.
  const multipleSelectFolders = ["models", "metrics"]

  // Convert folderContent into array.
  const newoptions = extractSecondLevelKeys(folderContent)

  // Check if the folder name is within the specific folders array.
  const isMultipleFolder = multipleSelectFolders.includes(folder)
  
  return (
    <CustomSelect
      name = { folder }
      options = { newoptions }
      placeholder = "Select or start typing..."
      isMulti = { isMultipleFolder }
    />
  )
}