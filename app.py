import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed
import torch

# -------------------------
# Config / model selection
# -------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
GEN_MODEL = "google/flan-t5-base"

# -------------------------
# Helpers: load pipelines
# -------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_name=SENTIMENT_MODEL):
    """Loads the sentiment analysis pipeline."""
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )

@st.cache_resource(show_spinner=False)
def load_generator(model_name=GEN_MODEL):
    """Loads the text generation model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


# Load models
with st.spinner("Loading AI models... This may take a moment."):
    sentiment_pipe = load_sentiment_pipeline()
    tokenizer, generator = load_generator()

# -------------------------
# Utility functions
# -------------------------
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

def detect_sentiment(prompt_text):
    """Detects the sentiment of a given text."""
    raw = sentiment_pipe(prompt_text)[0]
    top = max(raw, key=lambda x: x['score'])
    raw_label = top['label']
    score = float(top['score'])
    human_label = LABEL_MAP.get(raw_label, raw_label.lower())
    return human_label, score


def build_generation_prompt(user_prompt, sentiment_label):
    """Builds a detailed prompt for the generation model."""
    if sentiment_label == "positive":
        mood = "in an optimistic and uplifting tone"
    elif sentiment_label == "negative":
        mood = "in a critical and serious tone"
    else:
        mood = "in a balanced, factual tone"

    prompt = f"{user_prompt}\nWrite one paragraph {mood} about the above:"
    return prompt


def generate_text(
    prompt_text,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    required_sentiment: str = None,
    max_attempts: int = 3,
    min_keyword_matches: int = 0
):
    """
    Generate using a seq2seq model (Flan/T5). Optionally enforce sentiment + keyword checks.
    """
    set_seed(seed)

    def keywords_from_prompt(text, max_k=3):
        """Extracts simple keywords from the prompt."""
        toks = [t.strip(".,!?;:()[]\"'").lower() for t in text.split() if len(t) > 2]
        seen, out = set(), []
        for t in toks:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= max_k:
                break
        return out

    keywords = keywords_from_prompt(prompt_text, 3)
    last_out, sentiment_meta = "", None
    
    kw_matches = 0

    for attempt in range(1, max_attempts + 1):
        do_sample = True if temperature > 0 else False

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(generator.device)
        out = generator.generate(
            input_ids,
            max_new_tokens=min(max_new_tokens, 512),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        if gen.lower().startswith("paragraph:"):
            gen = gen[len("paragraph:"):].strip()
        gen = gen.split("\n\n")[0].strip()
        last_out = gen

        kw_matches = sum(1 for k in keywords if k and k in gen.lower())
        topical_ok = (kw_matches >= min_keyword_matches) if keywords else True

        sentiment_ok = True
        if required_sentiment:
            try:
                sraw = sentiment_pipe(gen)[0]
                top = max(sraw, key=lambda x: x['score'])
                raw_label = top['label']
                score = float(top['score'])
                human_label = LABEL_MAP.get(raw_label, raw_label.lower())
                sentiment_meta = {"label": human_label, "score": score}
                sentiment_ok = (human_label == required_sentiment.lower())
            except Exception:
                sentiment_ok = True  # fail-open

        if topical_ok and sentiment_ok:
            meta = {
                "attempt": attempt, "keywords": keywords,
                "keyword_matches": kw_matches, "sentiment": sentiment_meta
            }
            return gen, meta

        temperature = min(1.2, temperature + 0.2)

    meta = {
        "attempt": max_attempts, "keywords": keywords,
        "keyword_matches": kw_matches, "sentiment": sentiment_meta,
        "failure_reason": "Could not meet all generation criteria."
    }
    return last_out, meta


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentient Pen", layout="centered")

st.title("Sentient Pen - AI Sentiment Aligned Text Generator")

st.markdown("""
Enter a short prompt. The system will detect its sentiment (positive, neutral, or negative)
and generate a paragraph that matches that sentiment. You can also manually choose a
sentiment or adjust the generation settings in the sidebar.
""")

with st.sidebar:
    st.header("Generation Settings")
    max_len = st.slider("Max Length (tokens)", 40, 300, 120, 10)
    temp = st.slider("Temperature (creativity)", 0.0, 1.2, 0.4, 0.1)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05)
    seed = st.number_input("Random Seed", 0, 99999, 42)
    st.header("Model Info")
    st.write(f"**Sentiment:** `{SENTIMENT_MODEL}`")
    st.write(f"**Generator:** `{GEN_MODEL}`")

prompt = st.text_area("Enter your prompt here:", value="The campus library at dawn", height=100)
col1, col2 = st.columns([1, 1.5])
with col1:
    st_button = st.button("Generate Text", use_container_width=True)
with col2:
    manual_override = st.selectbox(
        "Override sentiment (optional):",
        ["Auto-detect", "positive", "neutral", "negative"]
    )

if "history" not in st.session_state:
    st.session_state.history = []

if st_button and prompt.strip():
    with st.spinner("Analyzing sentiment and generating text..."):
        auto_sentiment, conf = detect_sentiment(prompt)
        
        sentiment_to_use = manual_override if manual_override != "Auto-detect" else auto_sentiment
        
        final_prompt = build_generation_prompt(prompt, sentiment_to_use)

        generated, gen_meta = generate_text(
            prompt_text=final_prompt,
            max_new_tokens=max_len,
            temperature=temp,
            top_p=top_p,
            seed=seed,
            required_sentiment=sentiment_to_use,
            max_attempts=3,
            min_keyword_matches=1
        )

    st.markdown("### Generated Text")
    
    info_cols = st.columns(2)
    info_cols[0].info(f"**Detected Sentiment:** {auto_sentiment} ({conf:.2f})")
    info_cols[1].success(f"**Used Sentiment:** {sentiment_to_use}")

    st.markdown(f"> {generated}")
    
    with st.expander("Show Generation Details"):
        st.write(gen_meta)
    
    st.session_state.history.insert(
        0,
        {"prompt": prompt, "sentiment": sentiment_to_use, "out": generated}
    )
    if len(st.session_state.history) > 50:
        st.session_state.history.pop()

if st.session_state.history:
    st.markdown("---")
    st.markdown("### Recent Generations")
    for i, item in enumerate(st.session_state.history[:5]):
        with st.container():
            st.write(f"**Prompt:** *{item['prompt']}*")
            st.write(item['out'])
            if i < len(st.session_state.history) - 1 and i < 4:
                st.markdown("---")

