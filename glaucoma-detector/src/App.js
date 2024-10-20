import React from 'react';
import './App.css';

function App() {
  const handleClick = () => {
    alert("Button Clicked!");
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the Glaucoma Prediction App</h1>
        <p>
          This application uses machine learning models to predict the probability of glaucoma based on input images.
        </p>
        <button onClick={handleClick} className="App-button">
          Get Started
        </button>
      </header>
    </div>
  );
}

export default App;
