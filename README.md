The problem domain of this project lies at the intersection of Natural Language Processing (NLP), Information Retrieval, and Ethical Computing in Academia and Content Creation. Specifically, the domain addresses plagiarism detection — a sub-domain of text mining and NLP — where the goal is to identify instances of copied, rephrased, or semantically equivalent text that has been sourced from external references without proper citation.

Key areas within the problem domain include:
1. Text Similarity Detection
This involves comparing two or more bodies of text to determine how lexically or semantically similar they are. Both statistical models (TF-IDF, cosine similarity) and deep learning-based models (BERT embeddings) are employed.

2. Semantic Textual Analysis
Goes beyond keyword or phrase matching to analyze the meaning behind sentences, helping detect paraphrased plagiarism that traditional matchers cannot catch.

3. Web-Based Source Mining
Uses web search APIs (e.g., SerpAPI) to pull real-time external content from the internet. This extends detection beyond static academic databases, allowing up-to-date and more exhaustive comparison.

4. Information Extraction from Documents
Document handling (from .docx, .pdf, and raw text) using file parsers is necessary to process various forms of student and research submissions.

5. Academic Integrity and Content Verification
This project also contributes to educational tools designed to foster originality, fair evaluation, and ethical use of information.

Technologies and Concepts Used in the Problem Domain:
1. NLP Libraries: nltk, scikit-learn, sentence-transformers
2. File Parsers: python-docx, PyPDF2
3. Web APIs: serpapi for Google search result extraction, S
4. Streamlit for interactive frontend
5. Similarity Metrics: Cosine similarity, BERT embedding vectors

Workflow
1. Input Extraction:
.docx via python-docx
.pdf via PyPDF2
Plain text via Streamlit input box

2. Query Construction:
Use top 300 characters from user text as a search query to fetch Google results using SerpAPI.
Extract up to 10 snippets.
Similarity Calculation:
TF-IDF Cosine Similarity:
Preprocess and vectorize text.
Compute cosine similarity between input and web snippets.

3. BERT Semantic Similarity:
Generate sentence embeddings.
Compute cosine similarity using sentence-transformers.
Sentence-Level Analysis:
Split input text into sentences using NLTK Punkt tokenizer.
Calculate similarity score of each sentence with all web snippets.
Flag high-similarity sentences.

4. Output:
Display total plagiarism percentage (TF-IDF and BERT).
Show color-coded, sentence-level plagiarism detection.
Display source snippet scores.
