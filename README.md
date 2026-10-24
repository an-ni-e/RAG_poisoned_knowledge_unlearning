# RAG_poisoned_knowledge_unlearning

## Project Overview

This project investigates how the effectiveness of machine unlearning methods can be evaluated in Retrieval-Augmented Generation (RAG) systems containing polluted or poisoned knowledge. It focuses on whether behavioural unlearning can reduce the influence of poisoned information while preserving access to legitimate knowledge.

## Code and Implementation

This project involves code and is implemented in a Python-based Google Colab notebook. The notebook creates a controlled RAG environment using a synthetic knowledge base containing legitimate and poisoned documents, retrieves relevant documents for a set of queries, and evaluates the generated answers under different experimental conditions.

The experiments compare a clean knowledge base, a poisoned knowledge base, removal of poisoned documents, and two behavioural unlearning adaptations. The implementation uses an embedding model for retrieval and a language model to generate responses.

## Scope and Limitations

The project evaluates behavioural unlearning in a controlled RAG environment. It does not implement or evaluate permanent deletion of knowledge from a model’s internal weights. The findings are limited to the selected model, synthetic dataset, retrieval configuration and experimental conditions.

## Reproducibility

The notebook documents the experimental configuration and should be run in an environment with the required Python packages, model access, and computational resources. Results may vary across software versions or hardware configurations. Refer to the notebook and accompanying results files for the implementation details and experimental outputs.
