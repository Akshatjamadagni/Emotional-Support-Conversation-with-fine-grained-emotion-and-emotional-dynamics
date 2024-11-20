# Model  Frontend

This is a React-based frontend application that interfaces with a Flask backend server to generate responses using the cauESC model. The project was created with Create React App.

## Features

- Interactive user interface for model interaction
- Real-time response generation
- Connection to Flask backend server
- Modern React components and practices

## Prerequisites

Before running this project, make sure you have the following installed:
- Node.js (v14.0.0 or higher)
- npm (v6.0.0 or higher)
- Access to the cauESC model backend server

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [your-project-name]
```

2. Install dependencies:
```bash
npm install
```

3. Configure the backend connection:
   - Open `.env` file in the root directory
   - Set your backend URL:
```
REACT_APP_API_URL=http://localhost:5000
```

## Available Scripts

### `npm start`
- Runs the app in development mode
- Open [http://localhost:3000](http://localhost:3000) to view it in your browser
- The page will automatically reload when you make changes
- Any lint errors will appear in the console

### `npm test`
- Launches the test runner in interactive watch mode

### `npm run build`
- Builds the app for production to the `build` folder
- Bundles React in production mode and optimizes for best performance
- The build is minified and filenames include hashes
- Your app is ready for deployment

### `npm run eject`
**Warning: this is a one-way operation. Once you `eject`, you can't go back!**
- Ejects the build configuration for full control
- Copies all configuration files and dependencies
- Only use if you need custom configuration

## Project Structure

```
src/
├── components/       # React components           # API connection logic
├── utils/           # Utility functions
├── styles/          # CSS and styling files
└── App.js           # Main application component
```

## Connecting to Backend

The frontend communicates with the Flask backend through REST API calls. Example usage:

```javascript
const response = await fetch('${process.env.REACT_APP_API_URL}/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ prompt: userInput }),
});
```

## Development

1. Start the Flask backend server first
2. Run the frontend development server using `npm start`
3. Make API calls to `http://localhost:5000` (or your configured backend URL)

## Production Deployment

1. Build the project:
```bash
npm run build
```

2. Deploy the contents of the `build` folder to your hosting service

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

### Common Issues

1. Backend Connection Errors
   - Verify the Flask server is running
   - Check the API URL in `.env` file
   - Ensure CORS is properly configured

2. Build Failures
   - Clear npm cache: `npm cache clean --force`
   - Delete node_modules and reinstall: `rm -rf node_modules && npm install`

## Learn More

- [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started)
- [React documentation](https://reactjs.org/)

## License

[License] - See LICENSE file for details






