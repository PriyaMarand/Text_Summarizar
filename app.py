import re
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import gradio as gr



# --- Sentence Splitting ---
def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in parts if s.strip()]


# --- Extractive Summarizer ---
def summarize_extractive(text: str, max_sentences: int = 3, diversity: float = 0.7) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    # Sentence scores
    scores = X.sum(axis=1).A1
    ranked_idx = np.argsort(-scores)

    # Precompute cosine similarity
    sims = cosine_similarity(X)

    # Greedy selection
    selected, selected_idx = [], set()
    for idx in ranked_idx:
        if len(selected) >= max_sentences:
            break
        if not selected:
            selected.append((idx, sentences[idx]))
            selected_idx.add(idx)
        else:
            sim = max(sims[idx, [i for i, _ in selected]])
            if sim < diversity:
                selected.append((idx, sentences[idx]))
                selected_idx.add(idx)

    # Top up if needed
    if len(selected) < max_sentences:
        for idx in ranked_idx:
            if len(selected) >= max_sentences:
                break
            if idx not in selected_idx:
                selected.append((idx, sentences[idx]))

    # Keep original order
    selected.sort(key=lambda t: t[0])
    return " ".join([s for _, s in selected])


# --- Gradio App ---
def summarize_app(text, num_sentences, diversity):
    summary = summarize_extractive(text, max_sentences=num_sentences, diversity=diversity)

    # Unique filename for download
    filename = f"summary_{uuid.uuid4().hex[:8]}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary, filename


demo = gr.Interface(
    fn=summarize_app,
    inputs=[
        gr.Textbox(lines=8, placeholder="Paste your text here...", label="Input Text"),
        gr.Slider(1, 7, value=3, step=1, label="Number of Sentences"),
        gr.Slider(0.1, 0.9, value=0.7, step=0.1, label="Diversity"),
    ],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.File(label="Download Summary", file_types=[".txt"]),
    ],
    title="📝 Mini Text Summarizer",
    description="Paste a paragraph and get a concise extractive summary. Adjust sentence count & diversity. Download your summary as a text file."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)

