import axios from "axios";

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

// Export the handleSubmit function that also applies the nestJson transformation
export const handleSubmit = async (values) => {
  console.log("Submitted Values:", values);

  // Convert values to JSON string
  const jsonString = JSON.stringify(values, null, 2);
  // Parse it back to JSON object for transformation
  const jsonParsed = JSON.parse(jsonString);
  // Transform the JSON to nested structure
  const jsonTransformed = nestJson(jsonParsed);
  // Convert transformed JSON back to string for download
  const finalJsonString = JSON.stringify(jsonTransformed, null, 2);

  if (values.submitType === "training") {
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
  } else if (values.submitType === "testing") {
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
  } else if (values.submitType === "download") {
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
  }
};
