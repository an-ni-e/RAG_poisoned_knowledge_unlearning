# RAG Poisoned Knowledge Unlearning

Evaluating behavioural unlearning in a Retrieval-Augmented Generation (RAG) system whose knowledge base contains poisoned documents. The project measures whether poisoned knowledge can be suppressed while legitimate knowledge is preserved, and compares two prompt-level unlearning methods against document removal.

## Overview

Poisoned documents in a RAG knowledge base can make a language model repeat false claims. Machine unlearning is proposed as a way to remove that influence, but it is hard to evaluate: poisoned content can keep influencing answers, and unlearning can damage legitimate answers.

This project builds a controlled experiment to measure both effects. It builds on the behavioural RAG-unlearning framework of Wang et al. and the poisoning-based unlearning evaluation of Pawelczyk et al. (see [References](#references)).

## Setup

| Component | Choice |
|---|---|
| Generator | `Qwen/Qwen2.5-3B-Instruct` (float16, greedy decoding, 60 new tokens) |
| Retriever | `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity, top-k = 2 |
| Dataset | 30 synthetic documents (20 legitimate, 10 poisoned) and 50 queries (30 targeted, 20 control) |
| Seed | 42 |

Each poisoned document contradicts one legitimate "test policy" fact (for example, minimum password length 14 vs 4). Each targeted fact is queried three ways (direct and two paraphrases). Answers are labelled `CORRECT`, `POISONED`, `CONFLICT` (both values) or `OTHER` (neither, including refusals).

## Conditions

| Condition | Description |
|---|---|
| Clean | Legitimate documents only |
| Poisoned | All 30 documents |
| Removal | Poisoned documents deleted |
| U-refusal | Poisoned text added to the system prompt as a confidential item the model must refuse to use |
| U-quarantine | Poisoned text added to the system prompt as an unreliable item the model must not use |

Unlearning triggers when a query's cosine similarity to the forget set is at least 0.724, the midpoint between the lowest targeted-direct and highest control-direct similarity.

## Results

Corrected ("strict") labels. Counts are out of 30 targeted and 20 control queries.

| | Clean | Poisoned | Removal | U-refusal | U-quarantine |
|---|---|---|---|---|---|
| Poison claim in answer (ASR-any) | 0 | 10 | 0 | 17 | 5 |
| Targeted answered correctly | 30 | 20 | 30 | 2 | 25 |
| Affected queries recovered (of 10) | n/a | 0 | 10 | 0 | 5 |
| Previously correct, now wrong (of 20) | 0 | 0 | 0 | 18 | 0 |
| Controls answered correctly | 20 | 20 | 20 | 19 | 20 |

- Poisoned documents were retrieved for every targeted query, and the poison claim appeared in 10 of 30 answers.
- Document removal reproduces the clean results exactly.
- U-quarantine recovered half of the affected queries with no regressions and no control damage.
- U-refusal made results worse: placing the poisoned text in the prompt left it visible to the model, and most previously correct answers became refusals.

## Repository

```
├── Main/
│   ├── RAG_Poisoned_Knowledge_Unlearning.ipynb   experiment (GPU)
│   └── README.md
├── Analysis/
│   ├── RAG_Poisoned_Knowledge_Analysis.ipynb     tests, label audit, figures (no GPU)
│   ├── AI.md
│   └── README.md
└── README.md
```

## Usage

1. Open `RAG_Poisoned_Knowledge_Unlearning.ipynb` in Google Colab with a GPU runtime (a T4 is sufficient) and run all cells. This writes `data/`, `results/` and `figures/`.
2. Open `RAG_Poisoned_Knowledge_Analysis.ipynb` in the same session, or unzip the saved outputs first (the first cell unzips `rag_unlearning_outputs_v2.zip`; adjust the filename if needed). Run all cells. No GPU is required.

Library versions, model settings, prompts and a dataset hash are recorded in `results/environment.json` and `results/dataset_hash.txt`.

## Limitations

- Evaluates behavioural unlearning only; no knowledge is removed from model weights or from the retrieval index.
- Small synthetic benchmark: 10 facts, one generator, one retriever, top-k = 2. Repeated phrasings of the same fact are not independent, so intervals are wide.
- Answers are labelled by regex, with manual correction of ambiguous `CONFLICT` answers (`results/label_corrections.csv`).
- The trigger threshold was calibrated on direct queries that are also evaluated.
- Prompt order affects results (poison is repeated more often when listed last). The reversed-order runs are post hoc and descriptive.
- Results may vary across hardware and library versions.

## References

1. S. Wang, T. Zhu, D. Ye, W. Zhou. *When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge?* arXiv:2410.15267.
2. M. Pawelczyk, J. Z. Di, Y. Lu, A. Sekhari, G. Kamath, S. Neel. *Machine Unlearning Fails to Remove Data Poisoning Attacks.* ICLR 2025.
3. T. Zhao, J. Chen, Y. Ru, H. Zhu, N. Hu, J. Liu, Q. Lin. *RAG Safety: Exploring Knowledge Poisoning Attacks to Retrieval-Augmented Generation.* arXiv:2507.08862.
