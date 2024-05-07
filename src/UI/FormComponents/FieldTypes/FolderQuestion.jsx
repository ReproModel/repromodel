import React from "react";
import { Field } from "formik";

export function SmartFolderField({ folder, folderContent }) {
    // Define the array of specific folder names
    const multipleSelectFolders = ["models", "metrics"];

    // Check if the folder name is within the specific folders array
    const isMultipleFolder = multipleSelectFolders.includes(folder);
    return (
     <>
     
     {isMultipleFolder ? (
                // If folder is one of the specific ones, render this block
                Object.entries(folderContent).map(([file, fileContent]) => (
                    <div style={{ display: "flex", flexDirection: "column" }} key={file}> {/* Always use unique keys for list items in React */}
                        {Object.entries(fileContent).map(([className, fileDetails]) => (
                            <label key={className}> {/* Unique key for child list */}
                                <Field
                                    type="checkbox"
                                    name={folder}
                                    value={className}
                                />
                                {className}
                            </label>
                        ))}
                    </div>
                ))
            ) : (
                // If folder is not one of the specific ones, render an alternative component
                <Field key={folder} as="select" id={folder} name={folder}>
                  <option value="">Select an option</option>
                  {Object.entries(folderContent).map(([file, fileContent]) => (
                    <>
                      {" "}
                      {Object.entries(fileContent).map(
                        ([className, fileContent]) => (
                          <option value={className}>{className}</option>
                        )
                      )}
                    </>
                  ))}
                </Field>
            )}
     </>
    );
  }