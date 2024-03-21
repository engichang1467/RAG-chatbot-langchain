# RAG Chatbot with Langchain and HugginFace 🦜🔗🤗

Retrieval Augmented Generation Chatbot using Langchain 🦜🔗 and HuggingFace 🤗

## Overview

The concept of Retrieval Augmented Generation (RAG) involves leveraging pre-trained Large Language Models (LLM) alongside custom data to produce responses. This approach merges the capabilities of pre-trained dense retrieval and sequence-to-sequence models. In practice, RAG models first retrieve relevant documents, then feed them into a sequence-to-sequence model, and finally aggregate the results to generate outputs. By integrating these components, RAG enhances the generation process by incorporating both the comprehensive knowledge of pre-trained models and the specific context provided by custom data.

## Getting Started

### Environment Setup

To get started, create a virtual environment and activate it:

```bash
virtualenv venv
source venv/bin/activate
```

Create a local environment file (`.env`) and add your huggingface API key:

```bash
HF_TOKEN=your_huggingface_api_key
```

### Install Dependencies

Next, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Run the Application

Now, you can run the application:

```bash
gradio app.py
```

This will start the application, allowing you to chat with the RAG model.

## Usage

Once the application is up and running, you can interact with the chatbot through a web interface.

## Additional Resources

- Check out the chatbot on [![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mca183/retrieval-augmented-generation-langchain)
- Explore more about the databricks-dolly-15k dataset [here](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- Explore more about the Dynamic-TinyBERT model [here](https://huggingface.co/Intel/dynamic_tinybert)
- Explore more about the sentence-transformers (`all-MiniLM-L6-v2`) model [here](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
