import React from "react";
import {
  Stack,
  Paper,
  List,
  ListItem,
  ListItemText,
  Typography,
  Container,
} from "@mui/material";
import questions from "../choicesJSON/newQuestionsFormat.json";
import FlexibleBlock from "./ModuleStackComponents/FlexibleBlock";

function ModuleStack({ FormikProps }) {
  return (
    <Container maxWidth="md">
      <Typography variant="h4" align="center" gutterBottom>
        Your ReproModel Structure
      </Typography>
      <Stack spacing={2}>
        {Object.entries(questions).map(([folder, folderContent]) => (
          <>
            {
              FormikProps.touched && FormikProps.touched[folder] ? (
                <FlexibleBlock key={folder} status={"active"} name={folder} />
              ) : (
                <FlexibleBlock key={folder} status={"passive"} name={folder} />
              ) // This is the else case
            }
          </>
        ))}
      </Stack>
    </Container>
  );
}

export default ModuleStack;
