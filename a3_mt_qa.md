
# Part 3 - Open-Answer Questions



#### Q1: How is BERTScore calculated? Read the first three paragraphs in Section 3 -- called "Token representation", "Similarity Measure", and "BERTScore" -- in [this paper](https://arxiv.org/pdf/1904.09675.pdf) and give a technical description of how the BERTScore precision/recall/f1 is calculated in ~6 sentences. You do not need to describe anything outside the scope of these specific paragraphs.

- BERTScore evaluates the quality of machine-generated text by comparing it to a reference human-generated textgenerated text.

- The reference sentence and the candidate sentence are tokenized into words or subwords, and each token is converted into a contextual embedding using a model like BERT. These embeddings take into account the surrounding context of each token in the sentence.

- BERTScore then computes the cosine similarity between each token in the candidate sentence and each token in the reference sentence. The cosine similarity measures how similar the vectors are in orientation.

- It finds the maximum similarity score for each token in the candidate sentence with tokens in the reference sentence (for precision), and vice versa (for recall). This is done through a greedy matching strategy, by pairing each token in one sentence with the most semantically similar token in the other.

- Precision is the average of these maximum similarity scores across all tokens in the candidate sentence. 

- Recall is calculated in the same way, but with respect to the reference sentence tokens.

- F1 score is the harmonic mean of precision and recall.



#### Q2: How is COMET trained and calculated? Read Section 2.4 -- "Translation Ranking Model" -- in [this paper](https://arxiv.org/pdf/2009.09025.pdf) and give a technical description in ~6 sentences.

- The COMET model is a translation ranking model that uses a cross-lingual encoder to produce embeddings for segments in a given tuple.

- This tuple includes a source sentence (s), a better-ranked hypothesis (h+), and a worse-ranked hypothesis (h-), with a reference translation (r).

- It optimizes the embedding space during the training such that the Euclidean distance between the source and reference and the worse hypothesis (h-) is greater by at least a margin ε than the distance to the better hypothesis (h+).

- It takes a single hypothesis during inference with its corresponding source and reference.

- The quality of the hypothesis is quantified by the harmonic mean between its distances to the source and the reference.

- This similarity score is then bounded between 0 and 1 by making it inversely proportional to the harmonic mean distance. 

- The closer the distances of the hypothesis to the source and reference, the higher the similarity score, which means a better translation quality.



#### Q3: Given your understanding of BLEU, BERTScore and COMET, how would you interpret the Kendall's Tau correlation results? Which ones are the least and most correlated? What is your hypothesis regarding the reasons behind the lowest correlation for one metric and the highest correlation in the other?

- **COMET** has the highest positive correlation (`0.29266`), indicating that it agrees with human judgment most frequently among these metrics. 
  This may be due to COMET's approach to evaluating translations by considering the semantics of the entire sentence and the ability to learn from human evaluations directly.
  
- **BERTScore-Precision** has the next highest correlation (`0.23907`), followed closely by **BERTScore-F1** (`0.23604`) and **BERTScore-Recall** (`0.23179`). 
  BERTScore's use of contextual embeddings likely helps it to capture semantic meanings more effectively than metrics based purely on n-gram overlap. The fact that precision, recall, and F1 are all similar suggests that BERTScore is balancing these aspects well in the context of this dataset.
  
- **BLEU** shows a lower correlation (`0.23276`) compared to COMET and BERTScore.
  This is possibly because BLEU primarily focuses on n-gram overlap, which may not capture the fluency and semantic qualities that human judges consider.

- The **BLEU-1** score, which measures unigram precision, has the lowest positive correlation (`0.17456`). This might be because unigram precision alone does not capture the complexity of the translation task, such as the correct order of words and their context within a sentence, which are more heavily weighted by human judges.

- In conclusion, the metrics that account for semantics and sentence-level context (like COMET and BERTScore) are more likely to agree with human judgment than those based on surface-level features like n-gram overlap (like BLEU). This shows the importance of context and meaning in translation quality evaluation, rather than just matching individual words or phrases.



#### Q4: Assume you have a large set of story beginnings and you would like to evaluate how well a model completes the stories. What problem would you run into with BLEU and COMET? Would the same disadvantages apply to BERTScore and why? Give your justification. Answer in ~6 sentences.

- When evaluating story completions, BLEU might not perform well because it relies on exact matches of n-gram between the candidate text and a set of reference texts.
  Stories are creative and allow for a wide variety of acceptable continuations, making exact matches a poor indicator of quality.

- COMET, as a metric based on machine translation quality estimation, could counter issues if it was trained on data that is not representative of creative story completion.
  It could potentially misjudge the coherence and creativity of a story's continuation if its training data emphasized different aspects of text.

- On the other hand, BERTScore may perform slightly better because it measures semantic similarity at a token level using contextual embeddings.
  This could allow it to better capture the meaning and flow of a story, even if the exact words differ from any reference.
  BERTScore is more focused on semantic quality than n-gram overlap, which is important for evaluating creative texts like stories where paraphrasing and stylistic diversity are common.
