# Gemini Dataset Generator

A Python application that uses Google's Gemini API to automatically generate high-quality datasets based on your descriptions. It offers a user-friendly web interface for creating, managing, and downloading custom datasets in various formats.

## Features

- Generate custom datasets based on your description
- Define your own columns and specify the number of rows
- Use templates for quick dataset creation
- Web search enhancement for more realistic data
- Save and manage your API key in browser local storage
- View dataset history and access previous datasets
- Advanced visualization options for data exploration
- Export datasets in multiple formats (CSV, JSON, Excel)
- Mobile-responsive interface with light/dark theme support

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Get your Gemini API key from [Google AI Studio](https://ai.google.dev/)

## Usage

1. Start the application:

```bash
python web_enhanced_app.py
```

2. Open your web browser and go to http://127.0.0.1:5000/
3. Enter your Gemini API key in the settings panel
4. Fill in the form with dataset details or select a template
5. Click "Generate Dataset"
6. Preview, visualize, and download your dataset

## Deployment

This application can be deployed to several cloud platforms:

### Heroku

1. Create a Heroku account and install the Heroku CLI
2. Navigate to your project directory and run:
   ```bash
   heroku login
   heroku create gemini-dataset-generator
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku master
   ```
3. Open your application with: `heroku open`

### PythonAnywhere

1. Create a PythonAnywhere account
2. Upload your files using their Files tab
3. Create a new web app, selecting Flask and the appropriate Python version
4. Configure your WSGI file to point to your application

### Railway

1. Create a Railway account and install their CLI
2. Navigate to your project directory and run:
   ```bash
   railway login
   railway init
   railway up
   ```

## How It Works

1. The application sends your dataset requirements to the Gemini API
2. If the response limit is reached, the application automatically sends follow-up requests
3. Data is processed, formatted, and stored for visualization and download
4. All user preferences and history are saved in the browser's local storage

## Customization

You can modify the maximum number of rows, templates, or adjust the UI by editing the respective files:
- `main.py`: Core application logic and API integration
- `templates/index.html` and `templates/result.html`: Web interface
