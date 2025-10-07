# ✒️ **Sentient Pen: AI-Powered Sentiment-Aligned Text Generator**

Sentient Pen is a smart writing assistant that analyzes the **sentiment of your text** and generates a new paragraph that matches that emotional tone.
Whether you want an **optimistic**, **neutral**, or **critical** continuation, this tool uses powerful NLP models to craft text that aligns with the sentiment.

---

## ✨ **Key Features**

* 🧠 **Automatic Sentiment Detection** – Instantly classifies your prompt as *positive*, *neutral*, or *negative* using a state-of-the-art model.
* 📝 **Sentiment-Aligned Generation** – Generates coherent paragraphs that match the detected or selected sentiment.
* 🎛️ **Manual Sentiment Override** – Don’t agree with the detection? Choose your preferred sentiment manually.
* ⚙️ **Customizable Settings** – Control text length, creativity (temperature), and nucleus sampling.
* 🕓 **Generation History** – Keeps a list of your recent prompts and their generated outputs.
* ⚡ **Fast & Lightweight** – Runs locally with minimal setup. GPU acceleration is supported if available.

---

## 🚀 **How It Works**

The app runs a **two-stage pipeline**:

1. **Sentiment Analysis**
   Your input is analyzed using
   `cardiffnlp/twitter-roberta-base-sentiment`
   to determine its emotional tone (positive, neutral, or negative).

2. **Sentiment-Aligned Text Generation**
   A tailored instruction prompt is built, and `google/flan-t5-base` generates a paragraph that reflects the chosen sentiment.
   If the output doesn't match the required tone, the app retries with adjusted creativity settings to maintain quality.

---

## 🛠️ **Setup and Installation**

Follow these steps to run the app locally:

### 1. **Clone the Repository**

```bash
git clone https://github.com/kratagya99/Sentient-Pen.git
cd Sentient-Pen
```

---

### 2. **Create a Virtual Environment**

It's strongly recommended to use a virtual environment to isolate dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3. **Install the Required Dependencies**

Install all necessary packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

> 💡 *If you have a CUDA-enabled GPU, installing the correct version of PyTorch will speed up the model significantly. The app automatically falls back to CPU if no GPU is detected.*

---

## ▶️ **Run the Application**

Once setup is complete, start the Streamlit app:

```bash
streamlit run app.py
```

The app will automatically open in your browser.
If not, go to 👉 [http://localhost:8501](http://localhost:8501)

---

## 💻 **Technology Stack**

* **Language:** Python 3.9+
* **Framework:** Streamlit
* **Machine Learning:** PyTorch, Hugging Face Transformers
* **Models:**

  * `cardiffnlp/twitter-roberta-base-sentiment` – Sentiment Analysis
  * `google/flan-t5-base` – Text Generation

---

## 📂 **Project Structure**

```
├── venv                  # Virtual environment (not pushed to Git)
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🤝 **Contributing**

Contributions are welcome!
If you'd like to improve Sentient Pen:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push your branch (`git push origin feature-branch`)
5. Create a Pull Request

---

## ✍️ **Author**

**Your Name**

* GitHub: [@Kratagya99](https://github.com/kratagya99)
* LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/kratagya-shrivastava-6947b3267/)

---

## ⭐ **Support**

If you found this project useful, consider giving it a **⭐ on GitHub** — it really helps boost visibility and encourages further development!
