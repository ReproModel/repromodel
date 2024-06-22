import choices from "../../../repromodel_core/choices.json"
import FlexibleBlock from "../ui/flexible-block/flexible-block"
import React from "react"

import { Container, Stack, Typography } from "@mui/material"


function RepromodelStructure({ FormikProps }) {

const optionsToShow = ["models", "preprocessing", "datasets", "augmentations", "metrics", "losses", "lr_schedulers", "monitor", "data_splits", "early_stopping"]
  
  return (
    <Container maxWidth = "md">
       
      <Typography variant = "h2" align = "center" gutterBottom style = { { fontWeight: "400" } } >Your ReproModel <br/>Pipeline</Typography>
      
      <Stack spacing = { 2 }>
        
        { Object.entries(choices).map(([folder, folderContent]) => (
          <>
          {optionsToShow.includes(folder) && (
            FormikProps.touched && FormikProps.touched[folder] ? (
              <FlexibleBlock key={folder} status="passive" name={folder} />   //set to passive right now from active since it doesn't work as intended. Set back when fixed
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