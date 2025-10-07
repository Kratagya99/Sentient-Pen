✒️ Sentient Pen: AI-Powered Sentiment-Aligned Text Generator
Sentient Pen is a smart writing assistant that analyzes the sentiment of your text and generates a new paragraph that aligns with that specific emotional tone. Whether you need an optimistic, critical, or neutral continuation of your thoughts, this tool helps you craft text with the right feel.

✨ Key Features
Automatic Sentiment Detection: Instantly analyzes your input prompt to determine if the sentiment is positive, neutral, or negative using a state-of-the-art NLP model.

Sentiment-Aligned Generation: Generates a coherent paragraph that continues your thought, carefully matching the detected (or manually chosen) sentiment.

Manual Override: Don't agree with the AI's sentiment detection? Manually select the sentiment you want the generated text to have.

Customizable Generation: Fine-tune the creative output with adjustable settings for text length, temperature (creativity), and top-p sampling.

Generation History: Keeps a running history of your recent prompts and the text generated from them.

🚀 How It Works
The application uses a two-stage pipeline powered by models from the Hugging Face Hub:

Sentiment Analysis: The initial prompt is first processed by cardiffnlp/twitter-roberta-base-sentiment, a powerful model fine-tuned for sentiment classification.

Text Generation: A new, more detailed prompt is constructed that instructs the generation model, google/flan-t5-base, to write a paragraph in a tone that matches the desired sentiment. The application includes a retry mechanism that adjusts the temperature to ensure the generated text meets the sentiment criteria.

🛠️ Setup and Installation
To run this project locally, follow these steps:

1. Clone the repository:

git clone [https://github.com/YOUR_USERNAME/sentient-pen.git](https://github.com/YOUR_USERNAME/sentient-pen.git)
cd sentient-pen

2. Create a virtual environment:
It's recommended to use a virtual environment to keep dependencies isolated.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install the required libraries:
The requirements.txt file contains all the necessary packages.

pip install -r requirements.txt

(Note: If you have a CUDA-enabled GPU, installing the correct version of PyTorch can significantly speed up the models. The app will default to CPU if no GPU is found.)

▶️ How to Run the App
Once you have completed the setup, you can run the Streamlit application with the following command:

streamlit run sentiment_app.py

The application will open in a new tab in your default web browser.

💻 Technology Stack
Language: Python 3.9+

Framework: Streamlit

Machine Learning: PyTorch, Hugging Face Transformers

Models:

cardiffnlp/twitter-roberta-base-sentiment

google/flan-t5-base

This project was created as a demonstration of how modern NLP models can be combined to create powerful and intuitive applications.