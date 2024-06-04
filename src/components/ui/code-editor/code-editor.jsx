import { highlight, languages } from "prismjs/components/prism-core";

import dedent from "dedent";
import Editor from "react-simple-code-editor";
import React from "react";
import axios from "axios";

import "./code-editor.css";
import "prismjs/components/prism-clike";
import "prismjs/components/prism-python";
import "prismjs/themes/prism.css";
import { handleDownload } from "../../../utils/download-helpers";
import { handleCustomScriptSubmit } from "../../../utils/json-helpers";
import { capitalizeFirstLetter } from "../../../utils/string-helpers";

const CodeEditor = ({ label }) => {
  const [code, setCode] = React.useState(
    dedent`
    Select the kind of custom script you want to create. 
    Make sure to have the backend up and running. 
    `
  );
  const fetchCustomScriptTemplate = async (type) => {
    try {
      const response = await axios.get(
        "http://127.0.0.1:5005/get-custom-script-template",
        {
          params: { type },
        }
      );
      console.log(response.data);
      setCode(response.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  return (
    <div>
      <div className="container">
        <div className="save-container">
          <label className="save-lable" htmlFor="save">
            What kind of Custom Script?
          </label>

          <select
            className="category-dropdown"
            id="CustomScriptCategories"
            name="save"
          >
            <option value="augmentations">augmentations</option>
            <option value="datasets">datasets</option>
            <option value="early_stopping">early_stopping</option>
            <option value="losses">losses</option>
            <option value="metrics">metrics</option>
            <option value="models">models</option>
            <option value="postprocessing">postprocessing</option>
            <option value="preprocessing">preprocessing</option>
          </select>

          <button
            onClick={() => {
              const selectElement = document.getElementById(
                "CustomScriptCategories"
              );
              const selectedValue = selectElement.value;
              fetchCustomScriptTemplate(selectedValue);
            }}
            className="save-button"
          >
            Load Template
          </button>
        </div>

        <div className="container-content">
          <div className="container-editor-area">
            <Editor
              className="container-editor"
              value={code}
              onValueChange={(code) => setCode(code)}
              highlight={(code) => highlight(code, languages.py)}
            />
          </div>
        </div>

        <div>
          <button
            type="submit"
            className="button"
            onClick={() => {
              const selectElement = document.getElementById(
                "CustomScriptCategories"
              );
              const selectedValue = selectElement.value;
              handleCustomScriptSubmit(
                code,
                `Custom${capitalizeFirstLetter(selectedValue)}Script`,
                selectedValue
              );
            }}
          >
            Submit
          </button>

          <button
            type="submit"
            className="button right-button"
            onClick={() => navigator.clipboard.writeText(code)}
          >
            Copy
          </button>

          <button
            type="submit"
            className="button right-button"
            onClick={() => {
              const selectElement = document.getElementById(
                "CustomScriptCategories"
              );
              const selectedValue = selectElement.value;
              handleDownload(code, `Custom${capitalizeFirstLetter(selectedValue)}Script`, "py");
            }}
          >
            Download
          </button>
        </div>
      </div>
    </div>
  );
};

export default CodeEditor;
