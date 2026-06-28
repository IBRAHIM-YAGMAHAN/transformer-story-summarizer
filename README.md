# transformer-story-summarizer

An end-to-end, hybrid NLP pipeline designed to bypass the context window limitations of traditional sequence-to-sequence models when processing long-form narrative texts. By extracting chronological anchors and rich metadata before abstractive generation, this system delivers coherent, human-like summaries of story chapters without losing the underlying plot structure.
------

## System Architecture

Long narrative texts often suffer from information loss or hallucinations when directly fed into large models. This repository implements a multi-stage **hybrid abstractive summarization** framework:

1. **Chronological Sampling**: Extracts meaningful text excerpts from the beginning, middle, and end of the document.
2. **Named Entity Recognition (NER)**: Extracts key characters, locations, and organizations using `Davlan/distilbert-base-multilingual-cased-ner-hrl`.
3. **Keyword & Keyphrase Co-Extraction**: Harnesses both **KeyBERT** (for semantic n-grams) and **TF-IDF** (for statistical importance) to build a dense vocabulary layer.
4. **Metadata Augmentation & Token Tokenization**: Merges the sampled fragments, entities, and keywords under a structured task prefix.
5. **Abstractive Core**: Passes the token list into a fine-tuned sequence-to-sequence **BART (`facebook/bart-large-cnn`)** model optimized with precise inference constraints (`length_penalty`, `no_repeat_ngram_size`, etc.).
