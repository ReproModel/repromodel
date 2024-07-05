import axios from "axios"

function flattenJson(nestedJson) {
  const result = {};

  function flatten(obj, prefix = "") {
    for (const key in obj) {
      const value = obj[key];
      const prefixedKey = prefix.length ? `${prefix}:${key}` : key;

      if (
        typeof value === "object" &&
        value !== null &&
        !Array.isArray(value)
      ) {
        flatten(value, prefixedKey);
      } else {
        result[prefixedKey] = value;
      }
    }
  }

  flatten(nestedJson);
  return result;
}

function validateJSON(jsonString) {
  try {
    JSON.parse(jsonString);
    return true;
  } catch (error) {
    return false;
  }
}

// Function to be triggered when the config file is uploaded
export const handleFileChange = (event, setFieldValue) => {
  const file = event.currentTarget.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      if (validateJSON(text)) {
        alert("JSON file checked and valid. Click to continue");
        const nestedJson = JSON.parse(text);
        const flatJson = flattenJson(nestedJson);
        Object.keys(flatJson).forEach((key) => {
          setFieldValue(key, flatJson[key], false); // Set field value without triggering validatio±n
        });

        {
          /** 
              // Optionally download the flattened json for easier troubleshooting

              const flatJsonString = JSON.stringify(flatJson, null, 2);
             
              // Create a Blob for download
              const blob = new Blob([flatJsonString], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = "flat_config.json";
              document.body.appendChild(link); // Append to the page
              link.click(); // Trigger the download

              // Clean up: remove the link and revoke the blob URL
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
              */
        }
      } else {
        alert("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  }
};

function nestJson(data) {
  const result = {};

  function assign(obj, keyPath, value) {
    let lastKeyIndex = keyPath.length - 1;
    for (let i = 0; i < lastKeyIndex; ++i) {
      let key = keyPath[i];
      if (!(key in obj)) obj[key] = {};
      obj = obj[key];
    }
    obj[keyPath[lastKeyIndex]] = value;
  }

  for (const key in data) {
    const value = data[key];
    const keyPath = key.split(":");
    assign(result, keyPath, value);
  }

  return result;

}

// Helper function to clean the config file before download
function removeUnwantedParts(obj) {
  // Keys to be removed
  const keysToRemove = ["type", "default", "undefined", "submitType"];

  // Recursive function to traverse and clean the object
  function clean(obj) {
    if (Array.isArray(obj)) {
      // If it's an array, clean each item
      obj.forEach(item => clean(item));
    } else if (typeof obj === 'object' && obj !== null) {
      // If it's an object, iterate through its keys
      for (const key in obj) {
        if (keysToRemove.includes(key)) {
          // Remove the key if it's in the keysToRemove array
          delete obj[key];
        } else {
          // Otherwise, continue cleaning nested objects
          clean(obj[key]);
        }
      }
    }
  }

  // Start the cleaning process
  clean(obj);
  return obj;
}

function filterParamsForExistingKeys(json) {
  // Function to filter params
  function filterParams(obj, listKey, paramsKey) {
    let validKeys = new Set();
    
    if (Array.isArray(obj[listKey])) {
      validKeys = new Set(obj[listKey]);
    } else if (typeof obj[listKey] === 'string') {
      validKeys.add(obj[listKey]);
    }

    const params = obj[paramsKey];
    if (params) {
      for (const key in params) {
        if (!validKeys.has(key)) {
          delete params[key];
        }
      }
    }
  }

  // List of key pairs to filter
  const filterKeys = [
    { listKey: "models", paramsKey: "models_params" },
    { listKey: "preprocessing", paramsKey: "preprocessing_params" },
    { listKey: "datasets", paramsKey: "datasets_params" },
    { listKey: "augmentations", paramsKey: "augmentations_params" },
    { listKey: "metrics", paramsKey: "metrics_params" },
    { listKey: "losses", paramsKey: "losses_params" },
    { listKey: "early_stopping", paramsKey: "early_stopping_params" },
    { listKey: "lr_schedulers", paramsKey: "lr_schedulers_params" },
    { listKey: "optimizers", paramsKey: "optimizers_params" }
  ];

  // Apply filtering for each key pair
  filterKeys.forEach(({ listKey, paramsKey }) => {
    filterParams(json, listKey, paramsKey);
  });
  return json
}

// Export the handleSubmit function that also applies the nestJson transformation
export const handleSubmit = async (values) => {

  const type = values.submitType
  const clean_values = removeUnwantedParts(values)
  console.log("Submitted Values:", clean_values);

  // Convert values to JSON string
  const jsonString = JSON.stringify(clean_values, null, 2);
  // Parse it back to JSON object for transformation
  const jsonParsed = JSON.parse(jsonString);
  // Transform the JSON to nested structure
  const jsonTransformed = nestJson(jsonParsed);
  // Remove params from options that are not selected anymore. 
  const cleanedJson = filterParamsForExistingKeys(jsonTransformed);
  // Convert transformed JSON back to string for download
  const finalJsonString = JSON.stringify(cleanedJson, null, 2);

  if (type === "training") {
    // Handle training submit action
    try {
      const response = await axios.post(
        "http://127.0.0.1:5005/submit-config-start-training",
        jsonTransformed
      );
      console.log("Script Output:", response.data);
    } catch (error) {
      const errorMessage = error.response
        ? JSON.stringify(error.response.data)
        : error.message;
      console.warn("Warning: Error running script:", errorMessage);
      alert("Warning: Error running script: " + errorMessage);
    }
  } else if (type === "stop-training") {
      // Handle stop training submit action
      console.log("Trying to kill training")
      const response = await axios.post(
        "http://127.0.0.1:5005/kill-training-process",
        jsonTransformed
      );
      try {
        const response = await axios.post(
          "http://127.0.0.1:5005/kill-training-process",
          jsonTransformed
        );
        console.log("Script Output:", response.data);
      } catch (error) {
        const errorMessage = error.response
          ? JSON.stringify(error.response.data)
          : error.message;
        console.warn("Warning: Error running script:", errorMessage);
        alert("Warning: Error running script: " + errorMessage);
      }
  } else if (type === "testing") {
    // Handle testing submit action
    try {
      const response = await axios.post(
        "http://127.0.0.1:5005/submit-config-start-testing",
        jsonTransformed
      );
      console.log("Script Output:", response.data);
    } catch (error) {
      const errorMessage = error.response
        ? JSON.stringify(error.response.data)
        : error.message;
      console.warn("Warning: Error running script:", errorMessage);
      alert("Warning: Error running script: " + errorMessage);
    }
  } else if (type === "stop-testing") {
      // Handle stop testing submit action.
      try {
        const response = await axios.post(
          "http://127.0.0.1:5005/kill-testing-process",
          jsonTransformed
        );
        console.log("Script Output:", response.data);
      } catch (error) {
        const errorMessage = error.response
          ? JSON.stringify(error.response.data)
          : error.message;
        console.warn("Warning: Error running script:", errorMessage);
        alert("Warning: Error running script: " + errorMessage);
      }
  } else if (type === "download") {
    // Handle download action
    // Create a Blob for download
    const blob = new Blob([finalJsonString], { type: "application/json" });

    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "config.json";
    document.body.appendChild(link); // Append to the page
    link.click(); // Trigger the download

    // Clean up: remove the link and revoke the blob URL
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } else {
    console.log("Undefined Type in Submit Type");
  }
}

// Export the handleSubmit function that also applies the nestJson transformation
export const handleCustomScriptSubmit = async (code, filename, type) => {

  // Create a blob from the code string
  const file = new Blob([code], { type: "text/plain" });

  // Create a FormData object
  const formData = new FormData();
  formData.append("file", file, filename);
  formData.append("filename", filename);
  formData.append("type", type);

  try {
    const response = await axios.post(
      "http://127.0.0.1:5005/save-custom-script",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    console.log("Script Output:", response.data);
    alert("Your new custom script was save to the backend location:\n\n" + response.data.path);
  } catch (error) {
    const errorMessage = error.response
      ? JSON.stringify(error.response.data)
      : error.message;
    console.warn("Warning: Error running script:", errorMessage);
    alert("Warning: Error running script: " + errorMessage);
  }
};