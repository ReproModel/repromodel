import "./code-extractor-modal.css"

import axios from "axios"
import CloseIcon from "@mui/icons-material/Close"
import React from "react"

import { Box, Button, Checkbox, FormControlLabel, IconButton, Modal, Radio, RadioGroup, Typography } from "@mui/material"
import { Form } from "formik"

const CodeExtractorModal = ({ open , handleClose }) => {
  
    // Text Input - Personal Access Token
    const [personalAccessToken, setPersonalAccessToken] = React.useState("")

    // Text Input - GitHub Repo Name
    const [repositoryName, setRepositoryName] = React.useState("")

    // Radio Buttons - GitHub Repo Visibility
    const [visibility, setVisibility] = React.useState("public")

    // Checkboxes - Confirmation
    const [ confirmationOne, setConfirmationOne ] = React.useState(false)
    const [ confirmationTwo, setConfirmationTwo ] = React.useState(false)

    // Extract Button onClick()
    const onSubmit = async (personalAccessToken, repositoryName, visibility) => { 
        
        console.log("Extract button clicked.")
        console.log("personalAccessToken: ", personalAccessToken)
        console.log("repositoryName: ", repositoryName)
        console.log("visibility: ", visibility)
    }

    return (
        <div className = "code-extractor-modal-container">
        
        <Modal open = { open }>
            
            <Box className = "code-extractor-modal">

                <Form>
            
                    {/* Close Button */}
                    <IconButton onClick = { handleClose } style = { { position: "absolute", right: 8, top: 8 } }>
                        <CloseIcon />
                    </IconButton>

                    {/* Title */}
                    <Typography variant = "h5" component = "h2" className = "code-extractor-modal-title">
                        <svg height = "19" width = "19" viewBox = "0 0 14 14">
                            <path d = "M7 .175c-3.872 0-7 3.128-7 7 0 3.084 2.013 5.71 4.79 6.65.35.066.482-.153.482-.328v-1.181c-1.947.415-2.363-.941-2.363-.941-.328-.81-.787-1.028-.787-1.028-.634-.438.044-.416.044-.416.7.044 1.071.722 1.071.722.635 1.072 1.641.766 2.035.59.066-.459.24-.765.437-.94-1.553-.175-3.193-.787-3.193-3.456 0-.766.262-1.378.721-1.881-.065-.175-.306-.897.066-1.86 0 0 .59-.197 1.925.722a6.754 6.754 0 0 1 1.75-.24c.59 0 1.203.087 1.75.24 1.335-.897 1.925-.722 1.925-.722.372.963.131 1.685.066 1.86.46.48.722 1.115.722 1.88 0 2.691-1.641 3.282-3.194 3.457.24.219.481.634.481 1.29v1.926c0 .197.131.415.481.328C11.988 12.884 14 10.259 14 7.175c0-3.872-3.128-7-7-7z" fill = "black" fillRule = "nonzero"/>
                        </svg>
                        <span style = { { paddingLeft: "8px" } }>Code Extractor</span>
                    </Typography>
                
                    {/* Subtitle */}
                    <div style = { { textAlign: "center" } }>
                        <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif"} }>Extract your code to a new GitHub repository.</span>
                    </div>
                    
                    {/* Text Input - Personal Access Token */}
                    <div>
                        <label className = "personal-access-token-container">
                            <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", marginLeft: "26px", fontSize: "12px" } }>Personal Access Token</span>
                            <input id = "personal-access-token-input" className = "personal-access-token-input" type = "text" onChange = { (e) => { setPersonalAccessToken(e.target.value) }} value = { personalAccessToken } />
                        </label>
                    </div>

                    {/* Text Input - GitHub Repository Name */}
                    <label className = "repository-name-input-container">
                        <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", marginLeft: "16px", fontSize: "12px" } } value = { repositoryName }>GitHub Repository Name</span>
                        <input id = "repository-name-input" className = "repository-name-input" type = "text" onChange = { (e) => { setRepositoryName(e.target.value) }} />
                    </label>
                
                    {/* Radio Button - GitHub Repo Visibility */}
                    <div className = "radio-repo-visibility-container">
                    
                        <FormControlLabel label = {<span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", marginLeft: "12px", fontSize: "12px" } }>Repository Visibility</span>} labelPlacement = "start" 
                            
                            control = {

                            <RadioGroup id = "radio-repo-visibility" className = "radio-repo-visibility" onChange = { (e, value) => { setVisibility(value) }} sx = { { marginLeft: "34px"} } row>
                                                    
                                <FormControlLabel value = "public" control = { <Radio checked = { visibility == "public" } style = { { opacity: "50%", color: "black", fontSize: "12px", root: { "& .MuiSvgIcon-root": { height: "1.5em", width: "2.5em" } } } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Public</span> } />

                                <FormControlLabel value = "private" control = { <Radio checked = { visibility == "private" } style = { { opacity: "50%", color: "black", fontSize: "12px" } }/> } label = { <span style = { { fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif", fontSize: "12px" } }>Private</span> } />

                            </RadioGroup>

                        }/>
                                   
                    </div>

                    {/* Checkbox - Confirmation */}
                    <div className = "code-extractor-confirmation">
                        <FormControlLabel
                            control = {<Checkbox onChange = { (e, value) => { setConfirmationOne(value) }} style = { { transform: "scale(.5)", padding: "0", marginLeft: "24px", pointerEvents: "none" } }/>}
                            label = { <Typography style = { { opacity: "50%", fontSize: "10px", textAlign: "left", padding: "0" } }>I understand that the previous experiment code will be deleted in order to generate a new one.</Typography> }
                            sx = {{ fontSize: "4px"}}
                        />
                        <FormControlLabel
                            control = {<Checkbox onChange = { (e, value) => { setConfirmationTwo(value) }}  style = { { transform: "scale(.5)", padding: "0", marginLeft: "24px", pointerEvents: "none" } }/>}
                            label = { <Typography style = { { opacity: "50%", fontSize: "10px", textAlign: "left", padding: "0" } }>I also confirm that the training process has been finished.</Typography> }
                            sx = {{ fontSize: "4px"}}
                        />         
                    </div>

                    {/* Button - Extract */}
                    <div className = "code-extractor-modal-button">
                        <Button variant = "contained" onClick = { () => onSubmit(personalAccessToken, repositoryName, visibility) } style = { { marginTop: "12px" } } disabled = { !confirmationOne || !confirmationTwo }>
                            <svg height = "14" width = "14" viewBox = "0 0 14 14">
                                <path d = "M7 .175c-3.872 0-7 3.128-7 7 0 3.084 2.013 5.71 4.79 6.65.35.066.482-.153.482-.328v-1.181c-1.947.415-2.363-.941-2.363-.941-.328-.81-.787-1.028-.787-1.028-.634-.438.044-.416.044-.416.7.044 1.071.722 1.071.722.635 1.072 1.641.766 2.035.59.066-.459.24-.765.437-.94-1.553-.175-3.193-.787-3.193-3.456 0-.766.262-1.378.721-1.881-.065-.175-.306-.897.066-1.86 0 0 .59-.197 1.925.722a6.754 6.754 0 0 1 1.75-.24c.59 0 1.203.087 1.75.24 1.335-.897 1.925-.722 1.925-.722.372.963.131 1.685.066 1.86.46.48.722 1.115.722 1.88 0 2.691-1.641 3.282-3.194 3.457.24.219.481.634.481 1.29v1.926c0 .197.131.415.481.328C11.988 12.884 14 10.259 14 7.175c0-3.872-3.128-7-7-7z" fill = "white" fillRule = "nonzero"/>
                            </svg>
                            <span style= { { marginLeft: "8px", marginTop: "4px", fontSize: "12px"} }>
                                Push
                            </span>
                        </Button>
                    </div>

                </Form>

            </Box>

        </Modal>

        </div>
    )
}

export default CodeExtractorModal