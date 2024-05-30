import { highlight, languages } from "prismjs/components/prism-core"

import dedent from "dedent"
import Editor from "react-simple-code-editor"
import React from "react"

import "./code-editor.css"
import "prismjs/components/prism-clike"
import "prismjs/components/prism-python"
import "prismjs/themes/prism.css"

const CodeEditor = ({ label }) => {
  
  const [code, setCode] = React.useState(
    dedent`
    import torch
    import torch.nn as nn
    from ..decorators import enforce_types_and_ranges

    class CustomModel(nn.Module):
        @enforce_types_and_ranges({
            'lr': {'type': float, 'default': 0.01, 'range': (0.0001, 1.0)},
            'activation': {'type': str, 'default': 'relu', 'options': ['relu', 'sigmoid', 'tanh']}
        })
        def __init__(self, lr, activation):
            super(CustomModel, self).__init__()
            self.lr = lr
            self.activation = activation
            # Define the layers of the model depending on the activation function perhaps
            pass
        
        def forward(self, x):
            # Define the forward pass using the activation function
            return x

    # Example of using the model:
    model = CustomModel(lr=0.005, activation='sigmoid')
    print("Model learning rate:", model.lr)
    print("Model activation function:", model.activation)
    `
  )

  return (
    <div>

      <div className = "container">

          <div className = "label">
            <strong>{ label }</strong>
          </div>

          <div className = "save-container">
            
            <label className = "save-lable" htmlFor = "save">Save Location:</label>
            
            <select className = "save-dropdown" id = "save" name = "save">
              <option value = "augmentations">augmentations</option>
              <option value = "datasets">datasets</option>
              <option value = "early_stopping">early_stopping</option>
              <option value = "losses">losses</option>
              <option value = "metrics">metrics</option>
              <option value = "models">models</option>
              <option value = "postprocessing">postprocessing</option>
              <option value = "preprocessing">preprocessing</option>   
            </select>
          
          </div>
        
          <div className = "container-content">
            
            <div className = "container-editor-area">
              
              <Editor
                className = "container-editor"
                value = { code }
                onValueChange = { code => setCode(code) }
                highlight = { code => highlight(code, languages.py) }
              />
            </div>
          
          </div>

          <div>

            <button
              type = "submit"
              className = "button"
              onClick = { () => navigator.clipboard.writeText(code) }>
              Copy
            </button>

            <button
              type = "submit"
              className = "button right-button"
              onClick = { () => console.log("Download clicked.") }>
              Download
            </button>

            </div>

        </div>

    </div>
  )
}

export default CodeEditor