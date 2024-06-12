import "./modality-options.css";
import React, { useState } from 'react';


const ModalityOptions = ({ options, cardOptions, onOptionClick, selectedOptions ,group}) => {
  return (
    <div className="card-deck card-break">
      {options && (
        <>
          {Object.entries(options).map(([optionName, optionContent]) => (
            <div
              className={`card ${selectedOptions.includes(optionName) ? 'selected' : ''}`}
              key={optionName}
              onClick={() => onOptionClick(group, optionName)}
            >
              <div className="row">
                <div className="card-image-container">
                  <div
                    className="card-image"
                    style={{ backgroundImage: "url('https://production-media.paperswithcode.com/icons/task/48d55b59-3af2-4a6d-a195-572f1d4a1867.jpg')" }}
                  />
                </div>
                <div className="card-title">
                  <p>{optionName}</p>
                </div>
              </div>
              <div className="card-body">
                <div className="card-subtitle">{optionName}</div>
              </div>
            </div>
          ))}
        </>
      )}
    
    
 {/** Old version that renders from array. */}
      {cardOptions?.map((card, idx) => (
        <div className="card" key={idx} onClick={() => setClicked(true)}>
          <div className="row">
            <div className="card-image-container">
              <div
                className="card-image"
                style={{ backgroundImage: "url('" + card.image + "')" }}
              />
            </div>

            <div className="card-title">
              <p>{card.label}</p>
            </div>
          </div>

          <div className="card-body">
            <div className="card-subtitle"> {card.numPapers}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ModalityOptions;
