import React, { useState } from 'react';
import axios from 'axios';
import { Button, TextField, Select, MenuItem, FormControl, InputLabel, Typography, Box, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

const interpretabilityMethods = [
  'Individual Conditional Expectation (ICE)',
  'Partial Dependence Plot (PDP)',
  'Surrogate Models',
  'Feature Importance',
];

export default function Interpretability() {
  const [selectedMethod, setSelectedMethod] = useState('');
  const [modelFile, setModelFile] = useState(null);
  const [dataFile, setDataFile] = useState(null);
  const [featureNames, setFeatureNames] = useState('');
  const [targetName, setTargetName] = useState('');
  const [results, setResults] = useState(null);
  const [plotImage, setPlotImage] = useState(null);

  // State variables for each tab
  const [iceState, setIceState] = useState({
    feature: '',
    gridResolution: 100,
    percentiles: '0.05,0.95',
    kind: 'individual',
    subsample: 1.0,
    randomState: '',
  });

  const [pdpState, setPdpState] = useState({
    pdpFeatures: '',
    pdpKind: 'average',
    pdpGridResolution: 100,
  });

  const [surrogateState, setSurrogateState] = useState({
    surrogateModelType: 'decision_tree',
    maxDepth: 5,
    nEstimators: 100,
    plotPerformance: false,
    sampleSize: 100,
  });

  const [fiState, setFiState] = useState({
    fiFeatureNames: '',
  });

  const handleMethodChange = (event) => {
    setSelectedMethod(event.target.value);
    setResults(null);
    setPlotImage(null);
  };

  const handleModelFileChange = (event) => {
    setModelFile(event.target.files[0]);
  };

  const handleDataFileChange = (event) => {
    setDataFile(event.target.files[0]);
  };

  const handleFeatureNamesChange = (event) => {
    setFeatureNames(event.target.value);
  };

  const handleTargetNameChange = (event) => {
    setTargetName(event.target.value);
  };

  // Event handlers for ICE tab
  const handleIceChange = (field) => (event) => {
    const value = field === 'subsample' ? parseFloat(event.target.value) : event.target.value;
    setIceState((prevState) => ({ ...prevState, [field]: value }));
  };

  // Event handlers for PDP tab
  const handlePdpChange = (field) => (event) => {
    const value = field === 'pdpGridResolution' ? parseInt(event.target.value) : event.target.value;
    setPdpState((prevState) => ({ ...prevState, [field]: value }));
  };

  // Event handlers for Surrogate Models tab
  const handleSurrogateChange = (field) => (event) => {
    const value = field === 'plotPerformance' ? event.target.value === 'true' : event.target.value;
    setSurrogateState((prevState) => ({ ...prevState, [field]: value }));
  };

  // Event handlers for Feature Importance tab
  const handleFiChange = (field) => (event) => {
    setFiState((prevState) => ({ ...prevState, [field]: event.target.value }));
  };

  const runInterpretability = async () => {
    if (!modelFile || !dataFile) {
      alert('Please upload both model and data files.');
      return;
    }

    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('data', dataFile);
    formData.append('method', selectedMethod);
    formData.append('feature_names', featureNames);
    formData.append('target_name', targetName);

    // Append method-specific parameters to formData based on the selected tab
    if (selectedMethod === 'Individual Conditional Expectation (ICE)') {
      formData.append('feature', iceState.feature);
      formData.append('grid_resolution', iceState.gridResolution);
      formData.append('percentiles', iceState.percentiles);
      formData.append('kind', iceState.kind);
      formData.append('subsample', iceState.subsample);
      formData.append('random_state', iceState.randomState);
    } else if (selectedMethod === 'Partial Dependence Plot (PDP)') {
      formData.append('pdp_features', pdpState.pdpFeatures);
      formData.append('pdp_kind', pdpState.pdpKind);
      formData.append('pdp_grid_resolution', pdpState.pdpGridResolution);
    } else if (selectedMethod === 'Surrogate Models') {
      formData.append('surrogate_model_type', surrogateState.surrogateModelType);
      formData.append('max_depth', surrogateState.maxDepth);
      formData.append('n_estimators', surrogateState.nEstimators);
      formData.append('plot_performance', surrogateState.plotPerformance);
      formData.append('sample_size', surrogateState.sampleSize);
    } else if (selectedMethod === 'Feature Importance') {
      formData.append('fi_feature_names', fiState.fiFeatureNames);
    }

    try {
      const response = await axios.post('http://localhost:5005/interpretability', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(response.data.results);
      setPlotImage(response.data.plot_image);
      
      if (selectedMethod === 'Feature Importance') {
        setResults(response.data.feature_importance);
      }
    } catch (error) {
      console.error('Error running interpretability:', error);
      alert('An error occurred while running interpretability.');
    }
  };

  const renderResults = () => {
    if (!results) return null;

    if (selectedMethod === 'Feature Importance') {
      return (
        <Paper elevation={3} sx={{ p: 3, mb: 4, overflow: 'auto', maxHeight: '400px' }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Feature Importance Results:
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Feature</TableCell>
                  <TableCell>Importance</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((row, index) => (
                  <TableRow key={index}>
                    <TableCell>{row.Feature}</TableCell>
                    <TableCell>{row.Importance}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      );
    }

    return (
      <Paper elevation={3} sx={{ p: 3, mb: 4, overflow: 'auto', maxHeight: '400px' }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Results for {selectedMethod}:
        </Typography>
        <pre>{JSON.stringify(results, null, 2)}</pre>
      </Paper>
    );
  };

  const renderPlot = () => {
    if (!plotImage) return null;

    const downloadImage = () => {
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${plotImage}`;
      link.download = 'interpretability_plot.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };

    return (
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Plot for {selectedMethod}:
        </Typography>
        <img src={`data:image/png;base64,${plotImage}`} alt="Interpretability Plot" style={{ maxWidth: '100%' }} />
        <Button variant="contained" onClick={downloadImage} style={{ marginTop: '16px' }}>
          Download Plot
        </Button>
      </Paper>
    );
  };

  return (
    <Box className="space-y-8">
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Interpretability Method</InputLabel>
          <Select value={selectedMethod} onChange={handleMethodChange}>
            <MenuItem value="">Select a method</MenuItem>
            {interpretabilityMethods.map((method) => (
              <MenuItem key={method} value={method}>
                {method}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {selectedMethod && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Box className="space-y-4">
            <Typography variant="subtitle1">Upload a trained model file (.pkl)</Typography>
            <input type="file" accept=".pkl" onChange={handleModelFileChange} />
            <Typography variant="subtitle1">Upload a data file (.csv)</Typography>
            <input type="file" accept=".csv" onChange={handleDataFileChange} />
            <br />
            <br />
            {/* Add input fields for method-specific parameters */}
            {selectedMethod === 'Individual Conditional Expectation (ICE)' && (
              <>
                <TextField
                  label="Feature"
                  value={iceState.feature}
                  onChange={handleIceChange('feature')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <TextField
                  label="Grid Resolution"
                  type="number"
                  value={iceState.gridResolution}
                  onChange={handleIceChange('gridResolution')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <TextField
                  label="Percentiles (comma-separated)"
                  value={iceState.percentiles}
                  onChange={handleIceChange('percentiles')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Kind</InputLabel>
                  <Select value={iceState.kind} onChange={handleIceChange('kind')}>
                    <MenuItem value="individual">Individual</MenuItem>
                    <MenuItem value="average">Average</MenuItem>
                    <MenuItem value="both">Both</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  label="Subsample"
                  type="number"
                  value={iceState.subsample}
                  onChange={handleIceChange('subsample')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <TextField
                  label="Random State"
                  value={iceState.randomState}
                  onChange={handleIceChange('randomState')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
              </>
            )}

            {/* Add input fields for PDP parameters */}
            {selectedMethod === 'Partial Dependence Plot (PDP)' && (
              <>
                <TextField
                  label="Features (comma-separated)"
                  value={pdpState.pdpFeatures}
                  onChange={handlePdpChange('pdpFeatures')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Kind</InputLabel>
                  <Select value={pdpState.pdpKind} onChange={handlePdpChange('pdpKind')}>
                    <MenuItem value="average">Average</MenuItem>
                    <MenuItem value="individual">Individual</MenuItem>
                    <MenuItem value="both">Both</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  label="Grid Resolution"
                  type="number"
                  value={pdpState.pdpGridResolution}
                  onChange={handlePdpChange('pdpGridResolution')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
              </>
            )}

            {/* Add input fields for surrogate model parameters */}
            {selectedMethod === 'Surrogate Models' && (
              <>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Surrogate Model Type</InputLabel>
                  <Select value={surrogateState.surrogateModelType} onChange={handleSurrogateChange('surrogateModelType')}>
                    <MenuItem value="decision_tree">Decision Tree</MenuItem>
                    <MenuItem value="linear_regression">Linear Regression</MenuItem>
                    <MenuItem value="random_forest">Random Forest</MenuItem>
                    <MenuItem value="svm">SVM</MenuItem>
                    <MenuItem value="knn">KNN</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  label="Max Depth"
                  type="number"
                  value={surrogateState.maxDepth}
                  onChange={handleSurrogateChange('maxDepth')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <TextField
                  label="Number of Estimators"
                  type="number"
                  value={surrogateState.nEstimators}
                  onChange={handleSurrogateChange('nEstimators')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Plot Performance</InputLabel>
                  <Select value={surrogateState.plotPerformance.toString()} onChange={handleSurrogateChange('plotPerformance')}>
                    <MenuItem value="true">Yes</MenuItem>
                    <MenuItem value="false">No</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  label="Sample Size"
                  type="number"
                  value={surrogateState.sampleSize}
                  onChange={handleSurrogateChange('sampleSize')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
              </>
            )}

            {/* Add input fields for feature importance parameters */}
            {selectedMethod === 'Feature Importance' && (
              <>
                <TextField
                  label="Feature Names (comma-separated)"
                  value={fiState.fiFeatureNames}
                  onChange={handleFiChange('fiFeatureNames')}
                  fullWidth
                  sx={{ mb: 2 }}
                />
              </>
            )}
          </Box>
          <Button
            variant="contained"
            onClick={runInterpretability}
            style={{ backgroundColor: '#38512f', marginTop: '24px' }}
            fullWidth
          >
            Run Interpretability
          </Button>
        </Paper>
      )}

      {results && renderResults()}
      {plotImage && renderPlot()}
    </Box>
  );
}
