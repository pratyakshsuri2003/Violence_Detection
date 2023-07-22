# Violence Detection App

This repository contains a Flask web application for detecting violence-related objects in images and videos. The app uses a custom YOLOv5 model trained to identify various objects related to violence, such as guns, knives, military vehicles, and more.

## Demo

![Demo](demo.gif)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/violence-detection-app.git
cd violence-detection-app
1. Create a virtual environment and activate it (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # For Windows, use: venv\Scripts\activate

2. Install the required dependencies:

```bash
pip install -r requirements.txt

## Usage

1. Start the Flask application:

```bash
python app.py

- Access the application in your web browser at http://localhost:2003/.
- Upload an image or video to the application to detect violence-related objects.

## API Endpoints

- The application provides the following API endpoints:

- GET /test: A test endpoint to check if the API is running.

- POST /detection: Endpoint to detect violence-related objects in images or videos. Accepts image and video files in the request.


