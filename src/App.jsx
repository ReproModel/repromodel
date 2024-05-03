import { useState } from "react";
import "./App.css";
import { Box, Grid, Typography } from "@mui/material";
import Header from "./UI/Header";
import { Formik } from "formik";

import ModuleStack from "./UI/ModuleStack";
import DynamicFormBuilder from "./UI/DynamicFormBuilder";
import NewDynamicFormBuilder from "./UI/NewDynamicFormBuilder";

function App() {
  const [count, setCount] = useState(0);

  return (
    <>
      <Header />

      <Grid container direction={"row"}>
        <Grid item xs={4} className="stackContainer">
          <Grid item className="stackFrame">
            <ModuleStack />
          </Grid>
        </Grid>
        <Grid item xs={8} className="questionairContainer">
          <Grid item className="questionairFrame">
            <NewDynamicFormBuilder />
          </Grid>
        </Grid>
      </Grid>
    </>
  );
}

export default App;
