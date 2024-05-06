// Function to check if the uploaded JSON is valid
export function validateJSON(text) {
  try {
    JSON.parse(text);
    return true; // valid JSON
  } catch (error) {
    return false; // invalid JSON
  }
}

// Function to be triggered when the config file is uploaded
export const handleFileChange = (event) => {
  const file = event.currentTarget.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      if (validateJSON(text)) {
        alert("Valid JSON file");
      } else {
        alert("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  }
};

// Function to handle form submission
export const handleSubmit = (values) => {
  console.log("Submitted Values:", values);

  const jsonString = JSON.stringify(values, null, 2);

  const blob = new Blob([jsonString], { type: "application/json" });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "config.json";
  document.body.appendChild(link); // Append to the page
  link.click(); // Trigger the download
  // Clean up: remove the link and revoke the blob URL
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
