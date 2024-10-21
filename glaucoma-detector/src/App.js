import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFiles(event.target.files);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFiles) {
      alert("Please upload an image or images.");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append('images', selectedFiles[i]);
    }

    try {
      const response = await fetch('http://localhost:5000/api/predict', { // Update this URL to your Flask API endpoint
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log(data); // Handle the response data accordingly

      alert(`Prediction results: ${JSON.stringify(data)}`);
    } catch (error) {
      console.error("Error uploading images:", error);
      alert("An error occurred while uploading images.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Glaucoma OCT Model</h1>
        <p>
          This application uses machine learning models to detect the presence of glaucoma based on input images.
        </p>
        
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileChange}
            className="App-file-input"
          />
          <button type="submit" className="App-button">
            Classify Images
          </button>
        </form>
      </header>
    </div>
  );
}

export default App;
