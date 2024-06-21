import "./extract-code.css"

import axios from "axios"
import React from "react"

import { Button, Checkbox, FormControlLabel, Radio, RadioGroup, Typography } from "@mui/material"
import { Form } from "formik"

const ExtractCode = ({}) => {

    // Text Input - Personal Access Token
    const [personalAccessToken, setPersonalAccessToken] = React.useState("")

    // Text Input - GitHub Repo Name
    const [repositoryName, setRepositoryName] = React.useState("")

    // Text Input - GitHub Repo Description
    const [repositoryDescription, setRepositoryDescription] = React.useState("")

    // Radio Buttons - GitHub Repo Visibility
    const [visibility, setVisibility] = React.useState("public")

    // Checkboxes - Confirmation
    const [ confirmationOne, setConfirmationOne ] = React.useState(false)
    const [ confirmationTwo, setConfirmationTwo ] = React.useState(false)

    // Extract Code
    const extractCode = async () => { 

        try {
            const response = await axios.post("http://localhost:5005/copy-covered-files")
            console.log("Successfully copied experiment-specific code and files to the extracted_code folder.")
            alert("Successfully copied experiment-specific code and files to the extracted_code folder.")

        } catch (error) {
            const error_msg = "Error:\n\n" + error.response.data.message.split(":")[1]
            console.error(error_msg)
            alert(error_msg)
        }
    } 

    // Push to GitHub
    const pushGithub = async (personalAccessToken, repositoryName, repositoryDescription, visibility) => { 
    
        console.log("Text Area - Personal Access Token: ", personalAccessToken)
        console.log("Text Area - Repository Name: ", repositoryName)
        console.log("Text Area - Repository Description", repositoryDescription)
        console.log("Radio Button - Visibility: ", visibility)

        if (personalAccessToken.trim() == "" && repositoryName.trim() == "") {
            alert("Required Fields:\n\nPlease enter your personal access token and a respository name for your new repo.")
            return
        }
        else if (personalAccessToken.trim() == "") {
            alert("Required Field:\n\nPlease enter your personal access token.")
            return
        }
        else if (repositoryName.trim() == "") {
            alert("Required Field:\n\nPlease enter a respository name for your new repo.")
            return
        }
    
        const formData = new FormData();
        formData.append('github_token', personalAccessToken);
        formData.append('repo_name', repositoryName);
        formData.append('description', repositoryDescription);
        formData.append('privacy', visibility);
    
        const response = await axios.post('http://127.0.0.1:5005/create-repo', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        .then((response) => {
          
            alert(response.data.message)
            return response.data
        })
        .catch((error) => {
            const error_msg = "Error:\n\n" + error.message
            console.error(error_msg)
            alert(error_msg)
          throw error
        })
      }

    return (
        <Form className = "tab-container">
            
            {/* Header - Extract Code */}
            <div className = "extract-code-header">
                <Typography variant = "h7" style = { { fontWeight: "600" } }>Extract Code</Typography>
            </div>

            {/* Subheader - Extract Code */}
            <div className = "extract-code-subheader">
                <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Extract code files to the directory: repromodel_core {' > '} extracted_code</span>
            </div>

            {/* Checkbox - Confirmation One */}
            <div className = "code-extractor-confirmation-one">
                <FormControlLabel
                    control = {<Checkbox onChange = { (e, value) => { setConfirmationOne(value) }} style = { { transform: "scale(.5)", padding: "0", marginLeft: "24px", pointerEvents: "none" } }/>}
                    label = { <Typography style = { { opacity: "70%", fontSize: "10px", textAlign: "left", padding: "0" } }>I understand that the previous experiment code will be deleted in order to generate a new one.</Typography> }
                    sx = {{ fontSize: "4px"}}
                />
            </div>
            
            {/* Checkbox - Confirmation Two */}
            <div className = "code-extractor-confirmation-two">
                <FormControlLabel
                    control = {<Checkbox onChange = { (e, value) => { setConfirmationTwo(value) }}  style = { { transform: "scale(.5)", padding: "0", marginLeft: "24px", pointerEvents: "none" } }/>}
                    label = { <Typography style = { { opacity: "70%", fontSize: "10px", textAlign: "left", padding: "0" } }>I also confirm that the training process has been finished.</Typography> }
                    sx = {{ fontSize: "4px"}}
                />         
            </div>

            {/* Button - Extract Code */}
            <div className = "code-extractor-button">
                <Button variant = "contained" onClick = { extractCode } style = { { marginTop: "12px", backgroundColor: "#38512f", opacity: (!confirmationOne || !confirmationTwo) ? "40%" : "90%" } } disabled = { !confirmationOne || !confirmationTwo }>
                    <span style= { { marginTop: "2px", fontSize: "12px", color: "white"} }>
                        Extract
                    </span>
                </Button>
            </div>

            {/* Header - GitHub */}
            <div className = "github-header">
                <Typography variant = "h7" style = { { fontWeight: "600" } }>Push to GitHub</Typography>
            </div>

            {/* Subheader - GitHub */}
            <div className = "github-subheader">
                <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Push extracted code to a new GitHub repository.</span>
            </div>

            {/* Text Input - Personal Access Token */}
            <div className = "personal-access-token-container">
                <label>
                    <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Personal Access Token: <span className = "req">*</span></span>
                    <input id = "personal-access-token-input" className = "personal-access-token-input" type = "text" onChange = { (e) => { setPersonalAccessToken(e.target.value) }} value = { personalAccessToken } />
                </label>
            </div>

            {/* Text Input - GitHub Repository Name */}
            <div className = "repository-name-input-container">
                <label>
                    <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } } value = { repositoryName }>Repository Name: <span className = "req">*</span></span>
                    <input id = "repository-name-input" className = "repository-name-input" type = "text" onChange = { (e) => { setRepositoryName(e.target.value) }} />
                </label>
            </div>

            {/* Text Input - GitHub Repository Description */}
            <div className = "repository-description-input-container">
                <label>
                    <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } } value = { repositoryDescription }>Repository Description</span>
                    <input id = "repository-description-input" className = "repository-description-input" type = "text" onChange = { (e) => { setRepositoryDescription(e.target.value) }} />
                </label>
            </div>

            {/* Radio Button - GitHub Repo Visibility */}
            <div className = "radio-repo-visibility-container">

                <FormControlLabel label = {<span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Repository Visibility</span>} labelPlacement = "start" 

                    control = {

                    <RadioGroup id = "radio-repo-visibility" className = "radio-repo-visibility" onChange = { (e, value) => { setVisibility(value) }} sx = { { marginLeft: "64px", marginTop: "4px" } } row>

                        <FormControlLabel value = "public" control = { <Radio checked = { visibility == "public" } style = { { opacity: "50%", color: "black", fontSize: "12px", root: { "& .MuiSvgIcon-root": { height: "1.5em", width: "2.5em" } } } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Public</span> } />

                        <FormControlLabel value = "private" control = { <Radio checked = { visibility == "private" } style = { { opacity: "50%", color: "black", fontSize: "12px" } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Private</span> } />

                    </RadioGroup>

                }/>

            </div>

            {/* Button - GitHub */}
            <div className = "github-button">
                <Button variant = "contained" onClick = { () => pushGithub(personalAccessToken, repositoryName, repositoryDescription, visibility) } style = { { marginTop: "12px", backgroundColor: "#38512f", opacity: "90%" } }>
                    <svg height = "14" width = "14" viewBox = "0 0 14 14">
                        <path d = "M7 .175c-3.872 0-7 3.128-7 7 0 3.084 2.013 5.71 4.79 6.65.35.066.482-.153.482-.328v-1.181c-1.947.415-2.363-.941-2.363-.941-.328-.81-.787-1.028-.787-1.028-.634-.438.044-.416.044-.416.7.044 1.071.722 1.071.722.635 1.072 1.641.766 2.035.59.066-.459.24-.765.437-.94-1.553-.175-3.193-.787-3.193-3.456 0-.766.262-1.378.721-1.881-.065-.175-.306-.897.066-1.86 0 0 .59-.197 1.925.722a6.754 6.754 0 0 1 1.75-.24c.59 0 1.203.087 1.75.24 1.335-.897 1.925-.722 1.925-.722.372.963.131 1.685.066 1.86.46.48.722 1.115.722 1.88 0 2.691-1.641 3.282-3.194 3.457.24.219.481.634.481 1.29v1.926c0 .197.131.415.481.328C11.988 12.884 14 10.259 14 7.175c0-3.872-3.128-7-7-7z" fill = "white" fillRule = "nonzero"/>
                    </svg>
                    <span style= { { marginLeft: "8px", marginTop: "4px", fontSize: "12px"} }>
                        Push
                    </span>
                </Button>
            </div>

        </Form>
    )
}

export default ExtractCode