import React from 'react';
import { Stack, Paper, List, ListItem, ListItemText, Typography, Container } from '@mui/material';
import moduleStackData from '../choicesJSON/moduleStack.json';
import questions from '../choicesJSON/newQuestionsFormat.json';
import FlexibleBlock from './ModuleStackComponents/FlexibleBlock';

function ModuleStack() {
  return (
    <Container maxWidth="md">
      <Typography variant="h4" align="center" gutterBottom>
        Your ReproModel Structure
      </Typography>
      <Stack spacing={2} >
      {Object.entries(questions).map(
                      ([folder, folderContent]) => (
                        <FlexibleBlock key={folder} status={"active"} name={folder}/>
                      )
                    )}
        
      </Stack>
    </Container>
  );
}

export default ModuleStack;
