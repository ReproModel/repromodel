import "./status-checks.css";

import axios from "axios";
import React from "react";
import { useEffect, useState } from "react";

const StatusCheck = () => {
  const [isBackendActive, setIsBackendActive] = useState(false);
  const [isTrainingInProgress, setIsTrainingInProgress] = useState(false);
  const [isTestingInProgress, setIsTestingInProgress] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      axios
        .get("http://127.0.0.1:5005/ping")
        .then((response) => {
          if (response.status === 200) {
            setIsBackendActive(true);
            setIsTrainingInProgress(response.data.trainingInProgress);
            setIsTestingInProgress(response.data.cvTestingInProgress || response.data.finalTestingInProgress);
          } else {
            setIsBackendActive(false);
            setIsTrainingInProgress(false);
            setIsTestingInProgress(false);
          }
        })
        .catch((error) => {
          setIsBackendActive(false);
          setIsTrainingInProgress(false);
          setIsTestingInProgress(false);
        });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (

    <>
      <div className="status-wrapper-position">
        <div
          className="blinking"
          style={{
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            backgroundColor: isBackendActive ? "green" : "red",
            marginRight: "10px",
            marginLeft: "14px",
          }}
        />
        <span>{isBackendActive ? "Backend Active" : "Backend Offline"}</span>
      </div>
      {isTrainingInProgress && (
        <div className="status-wrapper-position position2">
          <div
            className="blinking"
            style={{
              width: "10px",
              height: "10px",
              borderRadius: "50%",
              backgroundColor: "blue",
              marginRight: "10px",
              marginLeft: "14px",
            }}
          />
          <span>Training Runs</span>
        </div>
      )}
      {isTestingInProgress && (
        <div className="status-wrapper-position position2">
          <div
            className="blinking"
            style={{
              width: "10px",
              height: "10px",
              borderRadius: "50%",
              backgroundColor: "blue",
              marginRight: "10px",
              marginLeft: "8px",
            }}
          />
          <span>Testing Runs</span>
        </div>
      )}
    </>
  );
};


export default StatusCheck;
