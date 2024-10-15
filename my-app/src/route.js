import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // Ensure you import Router
import './App.css';
import Home from "./Components/chatbot"; // Your chatbot component
import Navbar from './Components/navbar'; // Your Navbar component
import Working from "./Components/working"; // Example additional component
import Dataset from "./Components/dataset"; // Example additional component
import About from "./Components/about"; // Example additional component

function App() {
    return (
        <Router> {/* Wrap your app with Router */}
            <Navbar />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/working" element={<Working />} />
                <Route path="/dataset" element={<Dataset />} />
                <Route path="/about" element={<About />} />
            </Routes>
        </Router>
    );
}

export default App;
