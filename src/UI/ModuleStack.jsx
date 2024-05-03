import React from 'react';
import { Stack, Paper, List, ListItem, ListItemText, Typography, Container } from '@mui/material';
import moduleStackData from '../choicesJSON/moduleStack.json';
import FlexibleBlock from './ModuleStackComponents/FlexibleBlock';

function ModuleStack() {
  return (
    <Container maxWidth="md">
      <Typography variant="h4" align="center" gutterBottom>
        Your ReproModel Structure
      </Typography>
      <Stack spacing={2} >
        {moduleStackData.map((block, index) => (
          <FlexibleBlock key={index} status={block.status} name={block.name}/>
        
        ))}
      </Stack>
    </Container>
  );
}

export default ModuleStack;
