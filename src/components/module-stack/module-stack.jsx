import FlexibleBlock from "../../UI/ModuleStackComponents/FlexibleBlock"
import questions from "../../choicesJSON/newQuestionsFormat.json"
import React from "react"

import { Container, Stack, Typography } from "@mui/material"

function ModuleStack({ FormikProps }) {
  
  return (
    <Container maxWidth = "md">
      
      <Typography variant = "h4" align = "center" gutterBottom>Your ReproModel Structure</Typography>
      
      <Stack spacing = { 2 }>
        
        { Object.entries(questions).map(([folder, folderContent]) => (
          <>
            {
              FormikProps.touched && FormikProps.touched[folder] ? (
                <FlexibleBlock key = { folder } status = { "active" } name = { folder } />
              ) : (
                <FlexibleBlock key = { folder } status = { "passive" } name = { folder } />
              )
            }
          </>
        ))}

      </Stack>

    </Container>
  )
}

export default ModuleStack
