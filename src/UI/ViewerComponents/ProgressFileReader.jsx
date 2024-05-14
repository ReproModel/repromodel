import React from "react";
import { useState, useEffect } from "react";
import axios from "axios";

const ProgressFileReader = ({ fileName }) => {
  const [fileContent, setFileContent] = useState("");

  useEffect(() => {
    const fetchFileContent = async () => {
      try {
        const response = await fetch(`logs/Training_logs/${fileName}`);
        const text = await response.text();
        setFileContent(text);
      } catch (error) {
        console.error("Error fetching the file:", error);
      }
    };

    fetchFileContent();
    const interval = setInterval(fetchFileContent, 1000); // Update every XX ms

    return () => clearInterval(interval); // Clean up the interval on component unmount
  }, [fileName]);

  return (
    <>
    <h4>Progress</h4>
    <div
      style={{
        width: "95%",
        marginTop: "16px",
        height: "55%",
        border: "2px solid #000",
        padding: "10px",
        overflow: "auto",
        borderRadius: "10px",
      }}
    >
      
      <pre>{fileContent}</pre>
    </div>
    </>
  );
};

export default ProgressFileReader;
