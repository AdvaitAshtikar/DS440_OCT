import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('Please upload a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.glaucoma_probability);
    } catch (error) {
      console.error('Error uploading the file:', error);
      alert('Error while making prediction');
    }
  };

  return (
    <div className="App">
      <h1>Glaucoma Detection</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Submit</button>
      </form>
      {prediction !== null && (
        <div>
          <h2>Prediction:</h2>
          <p>{`Glaucoma Probability: ${(prediction * 100).toFixed(2)}%`}</p>
        </div>
      )}
    </div>
  );
}

export default App;
