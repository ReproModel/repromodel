import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Button, TextField, Select, MenuItem, FormControl, InputLabel, Typography, Box, Paper, Grid } from '@mui/material';

const testCategories = {
  'Condition Tests': [
    'Shapiro-Wilk Test',
    'Kolmogorov-Smirnov Test',
    'Anderson-Darling Test',
    'Q-Q Plot',
    'Histogram',
    "Levene's Test",
    "Bartlett's Test",
    'Box Plot',
    "Mauchly's Test of Sphericity",
    'Greenhouse-Geisser Correction',
    'Scatter Plot',
    'Correlation Coefficient',
    'Durbin-Watson Test',
  ],
  'Parametric Tests': [
    'Paired t-test',
    'Two-sample t-test',
    'Analysis of Variance (ANOVA)',
    'Repeated Measures ANOVA',
    'Linear Regression Analysis',
    'F-test for comparing variances',
    'Z-test',
  ],
  'Non-parametric Tests': [
    'Wilcoxon Signed-Rank Test',
    'Mann-Whitney U Test',
    'Kruskal-Wallis H Test',
    'Friedman Test',
    "Spearman's Rank Correlation",
    'Chi-Square Test',
    'Sign Test',
    'Kolmogorov-Smirnov Test',
  ],
};

export default function StatisticalTests() {
  const [csvFile, setCsvFile] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedTest, setSelectedTest] = useState('');
  const [testParams, setTestParams] = useState({});
  const [testResults, setTestResults] = useState(null);
  const [plotImage, setPlotImage] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setCsvFile(file);
  };

  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
    setSelectedTest('');
    setTestParams({});
  };

  const handleTestChange = (event) => {
    setSelectedTest(event.target.value);
    setTestParams({});
  };

  const handleParamChange = (param, value) => {
    setTestParams((prevParams) => ({ ...prevParams, [param]: value }));
  };

  const getTestInputs = () => {
    switch (selectedTest) {
      case 'Shapiro-Wilk Test':
      case 'Kolmogorov-Smirnov Test':
      case 'Anderson-Darling Test':
        return (
          <TextField
            label="Significance Level"
            type="number"
            inputProps={{ step: 0.01, min: 0, max: 1 }}
            value={testParams.alpha || 0.05}
            onChange={(e) => handleParamChange('alpha', parseFloat(e.target.value))}
          />
        );
      case 'Paired t-test':
      case 'Two-sample t-test':
        return (
          <>
            <TextField
              label="Significance Level"
              type="number"
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              value={testParams.alpha || 0.05}
              onChange={(e) => handleParamChange('alpha', parseFloat(e.target.value))}
            />
            <FormControl fullWidth>
              <InputLabel>Alternative Hypothesis</InputLabel>
              <Select
                value={testParams.alternative || 'two-sided'}
                onChange={(e) => handleParamChange('alternative', e.target.value)}
              >
                <MenuItem value="two-sided">Two-sided</MenuItem>
                <MenuItem value="less">Less</MenuItem>
                <MenuItem value="greater">Greater</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      // Add more cases for other tests as needed
      default:
        return null;
    }
  };

  const runTest = async () => {
    if (!csvFile) {
      alert('Please upload a CSV file first.');
      return;
    }

    const csvData = await csvFile.text(); // Read the CSV content as text
    const payload = {
      csvData: csvData,
      test: selectedTest,
      params: testParams,
    };

    try {
      const response = await axios.post('http://localhost:5005/statistical_test', payload, {
        headers: { 'Content-Type': 'application/json' },
      });
      setTestResults(response.data.result);
      setPlotImage(response.data.plot || null); // Ensure this is the base64 image
    } catch (error) {
      if (error.response) {
        // Server responded with a status other than 200
        console.error('Error running statistical test:', error.response.data);
        alert(`Error: ${error.response.data.error || 'An error occurred while running the test.'}`);
      } else {
        // Network error or other issues
        console.error('Error running statistical test:', error.message);
        alert('Network error: Please check your connection or try again later.');
      }
    }
  };

  const renderTestResults = () => {
    if (!testResults) return null;

    const tableStyle = { borderCollapse: 'collapse', width: '100%' };
    const thStyle = { border: '1px solid #ddd', padding: '8px' };
    const tdStyle = { border: '1px solid #ddd', padding: '8px' };

    if (Array.isArray(testResults)) {
      return (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Model Name</th>
              {Object.keys(testResults[0]).filter(key => key !== 'model_name').map(key => (
                <th key={key} style={thStyle}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {testResults.map((result, index) => (
              <tr key={index}>
                <td style={tdStyle}>{result.model_name}</td>
                {Object.entries(result).filter(([key]) => key !== 'model_name').map(([key, value]) => (
                  <td key={key} style={tdStyle}>{JSON.stringify(value)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
    } else {
      return (
        <table style={tableStyle}>
          <tbody>
            {Object.entries(testResults).map(([key, value]) => (
              <tr key={key}>
                <td style={tdStyle} className="font-bold">{key}</td>
                <td style={tdStyle}>{JSON.stringify(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      );
    }
  };

  const renderPlots = () => {
    if (!plotImage) return null;

    if (Array.isArray(plotImage)) {
      return (
        <Grid container spacing={2}>
          {plotImage.map((plot, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle1" sx={{ mb: 1 }}>
                  Plot for {selectedTest} - Model {index + 1}:
                </Typography>
                <img src={`data:image/png;base64,${plot}`} alt={`Test Plot ${index + 1}`} style={{ maxWidth: '100%' }} />
              </Paper>
            </Grid>
          ))}
        </Grid>
      );
    } else {
      return (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Plot for {selectedCategory} - {selectedTest}:
          </Typography>
          <img src={`data:image/png;base64,${plotImage}`} alt="Test Plot" style={{ maxWidth: '100%' }} />
        </Paper>
      );
    }
  };

  return (
    <Box className="space-y-8">
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          ref={fileInputRef}
        />
        <Button
          variant="contained"
          onClick={() => fileInputRef.current.click()}
          style={{ backgroundColor: '#38512f' }}
        >
          Upload CSV File
        </Button>
        {csvFile && (
          <Typography variant="body1" style={{ marginTop: '16px' }}>
            Selected file: {csvFile.name}
          </Typography>
        )}
      </Paper>

      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Test Category</InputLabel>
          <Select value={selectedCategory} onChange={handleCategoryChange}>
            <MenuItem value="">Select a test category</MenuItem>
            {Object.keys(testCategories).map((category) => (
              <MenuItem key={category} value={category}>
                {category}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {selectedCategory && (
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>Statistical Test</InputLabel>
            <Select value={selectedTest} onChange={handleTestChange}>
              <MenuItem value="">Select a test</MenuItem>
              {testCategories[selectedCategory].map((test) => (
                <MenuItem key={test} value={test}>
                  {test}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
      </Paper>

      {selectedTest && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Box className="space-y-4" sx={{ mb: 3 }}>
            {getTestInputs()}
          </Box>
          <Button
            variant="contained"
            onClick={runTest}
            style={{ backgroundColor: '#38512f', marginTop: '24px' }}
            fullWidth
          >
            Run Test
          </Button>
        </Paper>
      )}

      {testResults && Object.keys(testResults).length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mb: 4, overflow: 'auto', maxHeight: '400px' }}>
          <Typography variant="h6" sx={{ mb: 2, fontSize: '0.9rem', overflowWrap: 'break-word' }}>
            Test Results for {selectedCategory} - {selectedTest}:
          </Typography>
          {renderTestResults()}
        </Paper>
      )}

      {plotImage && renderPlots()}
    </Box>
  );
}