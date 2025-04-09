# run_topic_model.py
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
from datetime import datetime
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm.auto import tqdm
tqdm.pandas()

# Command-line arguments
parser = argparse.ArgumentParser(description="Run BERTopic analysis on a folder of documents.")
parser.add_argument("--input_dir", required=True, help="Path to directory containing .md documents")
parser.add_argument("--output_dir", required=True, help="Path to save outputs")
args = parser.parse_args()

DATA_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# --- Utility Functions ---
def extract_date_from_filename(filename):
    date_match = re.search(r'^(\d{8})_', filename)
    if date_match:
        date_str = date_match.group(1)
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    return None

def load_documents(directory):
    documents, timestamps, filenames = [], [], []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                date = extract_date_from_filename(filename)
                if date and text.strip():
                    documents.append(text)
                    timestamps.append(date)
                    filenames.append(filename)
    return pd.DataFrame({
        'text': documents,
        'date': timestamps,
        'filename': filenames
    })

def preprocess_text(text):
    text = text.lower().replace("\n", " ").strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop 
              and not token.is_punct 
              and not token.like_num
              and token.lemma_ not in ENGLISH_STOP_WORDS 
              and len(token.lemma_) > 2]
    return " ".join(tokens)

def preprocess_dataframe(df):
    df['cleaned_text'] = df['text'].progress_apply(preprocess_text)
    return df

def initialize_bertopic():
    return BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        verbose=True,
        calculate_probabilities=True,
        nr_topics='auto'
    )

def generate_topics(topic_model, df):
    embeddings = topic_model.embedding_model.encode(df['cleaned_text'])
    topics, _ = topic_model.fit_transform(df['cleaned_text'], embeddings=embeddings)
    df['topic'] = topics
    topics_over_time = topic_model.topics_over_time(
        docs=df['cleaned_text'],
        topics=topics,
        timestamps=df['year']
    )
    return topic_model, df, topics_over_time, embeddings

def save_visualizations(topic_model, topics_over_time, df, embeddings=None, output_dir="outputs"):
    Path(output_dir).mkdir(exist_ok=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(f"{output_dir}/topic_info.csv", index=False)
    print("✅ Saved topic_info.csv")

    styled_info = (topic_info.style
                  .background_gradient(cmap='Blues')
                  .set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'), ('text-align', 'center')]}]))
    with open(f"{output_dir}/topic_info.html", "w", encoding="utf-8") as f:
        f.write(styled_info.to_html())
    print("✅ Saved topic_info.html")

    fig_time = topic_model.visualize_topics_over_time(topics_over_time)
    fig_time.write_html(f"{output_dir}/topics_over_time.html")
    print("✅ Saved topics_over_time plots (HTML)")

    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html(f"{output_dir}/topic_hierarchy.html")
    print("✅ Saved topic hierarchy plots (HTML)")

    if embeddings is not None:
        fig_docs = topic_model.visualize_documents(
            df['cleaned_text'], embeddings=embeddings, hide_annotations=True)
        fig_docs.write_html(f"{output_dir}/document_topics.html")
        print("✅ Saved document-topic scatter plots (HTML)")
    else:
        print("ℹ️ Embeddings not provided — skipping document-topic plot")

    topic_term_dir = f"{output_dir}/topic_terms"
    Path(topic_term_dir).mkdir(exist_ok=True)

    for topic_num in topic_info['Topic']:
        terms = topic_model.get_topic(topic_num)
        term_df = pd.DataFrame(terms, columns=["Term", "Score"])
        term_df.to_csv(f"{topic_term_dir}/topic_{topic_num}_terms.csv", index=False)

        plt.figure(figsize=(10, 6))
        term_df.head(10).sort_values("Score").plot.barh(x="Term", y="Score", color='skyblue')
        plt.title(f"Topic {topic_num} - Top Terms")
        plt.tight_layout()
        plt.savefig(f"{topic_term_dir}/topic_{topic_num}_terms.png")
        plt.close()

    print(f"✅ Saved term scores for {len(topic_info)} topics (CSV & PNG)")

    with open(f"{output_dir}/interpretation_guide.txt", "w", encoding="utf-8") as f:
        f.write("""HOW TO INTERPRET YOUR TOPIC MODELING RESULTS:

1. TOPIC INFO TABLE (topic_info.csv)
   - Topic -1: Outliers/documents not assigned to any topic
   - Count: Number of documents per topic
   - Name: Auto-generated label (customize with topic_model.set_topic_labels())
   - Representative Words: Most characteristic terms

2. TOPICS OVER TIME (topics_over_time.html)
   - X-axis: Time periods from your documents
   - Y-axis: Topic frequency
   - Spikes: Increased discussion of that topic
   - Look for emerging/disappearing trends

3. TOPIC HIERARCHY (topic_hierarchy.html)
   - Shows how broader topics split into subtopics
   - Vertical distance shows relationship strength

4. DOCUMENT CLUSTERS (document_topics.html)
   - Points: Individual documents
   - Colors: Topic assignments
   - Tight clusters indicate coherent topics

5. TERM SCORES (topic_terms/ folder)
   - Each topic's most representative words
   - Higher scores = more defining for the topic

6. TOPIC TERM ANALYSIS (topic_terms/ folder)
   - For each topic, you'll find:
     • topic_X_terms.csv  →  A table of top terms with importance scores
     • topic_X_terms.png  →  A bar chart of the top 10 most representative terms

   - How to read:
     • "Term" column lists the most defining words for the topic.
     • "Score" indicates how strongly each term represents the topic (higher = more important).
     • Horizontal bar chart gives a quick visual snapshot of the most relevant terms.

   - Use these to:
     • Label your topics meaningfully.
     • Understand what each topic captures.
     • Compare how different topics emphasize different terms.

   - Tip: Topic -1 usually contains outlier documents that didn’t fit any clear theme.

TIPS:
- Good topics have clear, specific terms
- Check documents in questionable topics
- Adjust nr_topics if you have too many/few topics
""")
    print("✅ Saved interpretation_guide.txt")
    print(f"\n🎉 All outputs saved to '{output_dir}' directory.")

# --- Main Execution ---
print(f"Loading documents from: {DATA_DIR}")
raw_df = load_documents(DATA_DIR)
print(f"Loaded {len(raw_df)} documents")

print("Preprocessing text...")
df = preprocess_dataframe(raw_df)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year.astype(str)

# Identify platform and document type
platform = Path(DATA_DIR).parts[-2] if len(Path(DATA_DIR).parts) >= 2 else "UnknownPlatform"
doc_type = Path(DATA_DIR).parts[-1]

print(f"\n🔍 Platform: {platform} | Document Type: {doc_type}")
print(f"📄 Number of documents: {len(df['cleaned_text'])}")

if len(df['cleaned_text']) < 5:
    print(f"⚠️ Skipping {platform} - {doc_type}: Not enough documents to model topics.")
else:
    print(f"✅ Proceeding with topic modeling for {platform} - {doc_type}...")
    bertopic_model, df_with_topics, topics_time, embeddings = generate_topics(initialize_bertopic(), df)

    print("Saving visualizations and analysis...")
    save_visualizations(
        topic_model=bertopic_model,
        topics_over_time=topics_time,
        df=df_with_topics,
        embeddings=embeddings,
        output_dir=OUTPUT_DIR
    )

