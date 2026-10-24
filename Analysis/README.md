# Further Analysis, Tests and Figures
This notebook re-analyses the results already produced from the previous notebook (it does not call the language model, so it runs in seconds, with or without a GPU). 

It adds what the first notebook lacked:

1. one shared set of helper functions (no duplicate definitions)
2. unit tests and data-integrity checks, so every number is verified before it is reported
3. an auditable label-correction table (label_auto is never overwritten)
4. metrics whose denominators are computed from the data, not typed in
5. a robustness check on the answer classifier
6. three figures: the main results with uncertainty, a per-fact map of what the poison and the unlearning do, and the threshold plot

To run this notebook:
Use the same Colab session as the original notebook (the data/ and results/ folders must exist), or upload rag_unlearning_outputs.zip, unzip it, and run all cells top to bottom. 
