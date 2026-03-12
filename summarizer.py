import json
import spacy
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load GiNZA (Optimized for Japanese Named Entities and Dependencies)
nlp = spacy.load("ja_ginza")

def extract_sar_features(sentences):
    """Extracts the 9 features from Table 8 [cite: 327-368]."""
    if not sentences: return np.array([])
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        tfidf_matrix = np.zeros((len(sentences), 1))
    
    conjunctions = ["だから", "それで", "ので", "なぜなら", "それゆえ", 
                    "したがって", "よって", "そして", "それから", "それに", 
                    "また", "かつ", "そのうえ", "おまけに", "それに加えて", 
                    "しかし", "でも", "けれども", "それなのに", "それにしても", 
                    "とはいえ", "逆に", "もっとも", "あるいは", "または", "それとも",
                    "もしくは", "さて", "ところで", "それでは", "それなら", "ついでに",
                    "というのは", "すなわち", "つまり", "例えば", "なお", "ちなみに", "それとも",
                    "よそに", "打ち切り", "ことに", "もしかすると", "結局", "いわゆる"]
    
    auxiliaries = ["れる", "られる", "せる", "させる", "ない", "ぬ", "たい", "ます", "た", "だ", "そうだ", 
                   "らしい", "ようだ", "べき", "う", "よう"]


    all_features = []
    max_len = max([len(s) for s in sentences])
    avg_len = np.mean([len(s) for s in sentences])

    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        f1 = 1.0 - (i / len(sentences)) # Position
        f2 = len(sent) / max_len if len(sent) > 10 else 0.1 # Length
        f3 = np.mean(tfidf_matrix[i]) if i < len(tfidf_matrix) else 0 # TF-IDF
        f4 = sum([t.head.i for t in doc]) / len(doc) if len(doc) > 0 else 0 # Dependency
        f5 = 1.0 if any(t.dep_ == 'ROOT' for t in doc) else 0.0 # Predicate
        f6 = f3 / (len(sent) / avg_len) if avg_len > 0 else 0 # BM25
        f7 = len(doc.ents) # Named Entity [cite: 351]
        f8 = sum(1 for word in conjunctions if word in sent) # Conjunction [cite: 361]
        f9 = sum(1 for word in auxiliaries if word in sent) # Auxiliary [cite: 364]
        all_features.append([f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return np.array(all_features)

def get_purified_sentence(content):
    if not content or len(content.strip()) < 5: return ""
    doc = nlp(content)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]
    if not sentences: return ""
    if len(sentences) == 1: return sentences[0]

    X = extract_sar_features(sentences)
    weights = np.array([0.5, 1.0, 1.5, 1.0, 1.0, 1.0, 3.0, 1.0, 3.0])  # Weight for the sentence
    scores = np.dot(X, weights)
    return sentences[np.argmax(scores)]

# --- Processing the whole JSONL ---
input_path = 'data_jp/jp_data.jsonl'
output_path = 'data_jp/jp_data_summarized.jsonl'

print(f"Starting purification of {input_path}...")

with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:
    
    line_count = 0
    for line in f_in:
        try:
            data = json.loads(line)
            content = data.get('content', '')
            
            # Step 1: Extract max(A) [cite: 211]
            data['summarize_content'] = get_purified_sentence(content)
            
            # Step 2: Write back to JSONL (preserving URL metadata)
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            line_count += 1
            if line_count % 100 == 0:
                print(f"Processed {line_count} rows...")
        except Exception as e:
            print(f"Error at line {line_count}: {e}")

print(f"Finished! Processed {line_count} rows. Saved to {output_path}")