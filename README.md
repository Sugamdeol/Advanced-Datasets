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
**THE MAIN FILES ARE IN MASTER BRANCH**
1. Clone or download this repository
2. Install required dependencies:

```bash
pip install flask pandas numpy matplotlib seaborn google-generativeai beautifulsoup4 requests
```

3. Get your Gemini API key from [Google AI Studio](https://ai.google.dev/)

## Usage

1. Start the application:

```bash
python main.py
```

2. Open your web browser and go to http://127.0.0.1:5000/
3. Enter your Gemini API key in the settings panel
4. Fill in the form with dataset details or select a template
5. Click "Generate Dataset"
6. Preview, visualize, and download your dataset
