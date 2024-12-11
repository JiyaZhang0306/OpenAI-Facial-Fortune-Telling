# OpenAI Facial Fortune Biscuits 

Welcome to the **OpenAI Facial Fortune-Telling** project! This web application combines real-time facial emotion analysis and AI-powered personalized responses to provide engaging and fun fortune-telling insights.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**OpenAI Facial Fortune-Telling** is a Flask-based web application that uses:

- **[Face++ API](https://console.faceplusplus.com/)** 

  Detecting face and analyzing facial emotions in real-time.

- **OpenAI GPT** 

  Based on current user's facial emotion feedback, generating personalized fortune-telling responses.

This project showcases the integration of real-time image processing with advanced AI models, making it an exciting blend of technology and creativity.

**Extra**: this project use web camera to capture real-time frame, and return feed back into FACE++ API to doing real-time face analyze.

---

## Features

- **Real-Time Facial Emotion Analysis**: Connect the Face++ API and Webcam to detect emotions like happiness, anger, sadness, 
- **AI-Powered Fortune-Telling**: Integrates OpenAI GPT (via Azure OpenAI) to provide custom responses based on detected emotions.
-  **Webcam Integration**: Captures live video and processes frames for emotion detection.
- **Interactive Dashboard**: Displays analyzed emotions and AI-generated fortune biscuits on web.

---

## Work Flow

1. **Facial Emotion Detection**:
   - Frames are captured from the webcam in real time.
   - The Face++ API processes these frames to detect facial emotions.

2. **AI Response Generation**:
   - Detected emotions are sent to OpenAI GPT via Azure.
   - The GPT model generates personalized "fortunes" based on emotional analysis.

3. **Dynamic Display**:
   - Emotions and the GPT response are displayed dynamically on a split-view dashboard.

---

## Setup

### Prerequisites

1. Python 3.8+ installed on your system.

2. API keys for:
   - [Azure OpenAI](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/)
   
     set **AZURE_KEY** and **AZURE_ENDPOINT** in virtual environment
   
     ```
     set AZURE_KEY={YOUR_AZURE_KEY}
     set AZURE_ENDPOINT={YOUR_AZURE_ENDPOINT}
     ```
   
   - [Face++](https://www.faceplusplus.com/)
   
     ```
     api_key={your_api_key}
     api_secret={your_api_secret}
     ```
   
3. Required libraries (installed via `pip`):
   - Flask
   
   - aiohttp
   
   - OpenAI
   
   - NumPy
   
   - OpenCV
   
     ```
     pip install -r requirement.txt
     ```
   
     

### Installation

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/JiyaZhang0306/OpenAI-Facial-Fortune-Telling.git
   cd OpenAI-Facial-Fortune-Telling

1. **Set Up a Virtual Environment**:

   ```
   viruralenv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**:

   - Save your Face++ API key in a file named `api_key.txt`.

   - Save your Face++ API secret in a file named `api_secret.txt`.

   - Configure your Azure OpenAI API key and endpoint as environment variables:

     ```
     set AZURE_KEY={YOUR_AZURE_KEY}
     set AZURE_ENDPOINT={YOUR_AZURE_ENDPOINT
     ```

------

## Usage

1. **Run the Application**:

   ```
   flask run
   ```

2. **Web App**: Open your browser and go to `http://127.0.0.1:5000`.

3. **Use the Application**:

   - Allow webcam access to enable real-time emotion analysis.
   - View the detected emotions and AI-generated responses on the dashboard.

------

## Project Structure

```
graphqlCopy codeOpenAI-Facial-Fortune-Telling/
├── app.py                     # Main Flask application
├── fortuneteller.py           # AI logic and API integrations
├── static/                    # Static assets (CSS, JS, images)
├── templates/                 # HTML templates for Flask
├── api_key.txt                # Face++ API key (not included)
├── api_secret.txt             # Face++ API secret (not included)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

------

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request.

------

## License

This project is licensed under the MIT License. See the LICENSE file for details.

------

## Acknowledgments

- **Face++ API** for facial emotion analysis.
- **Azure OpenAI** for GPT integration.
- The amazing Python and Flask communities for making such projects possible!

------

Enjoy the blend of technology and creativity! Feel free to share your thoughts or suggest improvements. 😊

```
sqlCopy code
### How to Use
1. Copy this into a new `README.md` file.
2. Update any missing sections, especially with your license details or additional setup instructions.
```