import choices from "../../../repromodel_core/choices.json"
import FlexibleBlock from "../flexible-block/flexible-block"
import React from "react"

import { Container, Stack, Typography } from "@mui/material"


function RepromodelStructure({ FormikProps }) {

const optionsToShow = ["models", "preprocessing", "datasets", "augmentations", "metrics", "losses", "lr_schedulers", "monitor", "data_splits"]
  
  return (
    <Container maxWidth = "md">
      
      <Typography variant = "h4" align = "center" gutterBottom>Your ReproModel Structure</Typography>
      
      <Stack spacing = { 2 }>
        
        { Object.entries(choices).map(([folder, folderContent]) => (
          <>
          {optionsToShow.includes(folder) && (
            FormikProps.touched && FormikProps.touched[folder] ? (
              <FlexibleBlock key={folder} status="active" name={folder} />
            ) : (
              <FlexibleBlock key={folder} status="passive" name={folder} />
            )
          )}
        </>
        ))}

      </Stack>

    </Container>
  )
}

export default RepromodelStructure