# Med-HoDV

Med-HoDV is a medical KG-RAG framework based on **medical hypothesis generation** and **dual-dimensional validation**.  
This repository implements the method described in Chapter 4 of the thesis: **a Medical KG-RAG Method Based on Medical Hypothesis and Dual-Dimensional Validation**.

The code is adapted from the HyKGE framework and reorganized to match the Med-HoDV pipeline.

## Overview

Med-HoDV introduces a medical-domain KG-RAG workflow consisting of five core modules:

1. **Med-HOM**: Medical Hypothesis Organization Module  
   Generates a structured medical hypothesis from the original question.

2. **Med-NM**: Medical Named Entity Module  
   Extracts medical entities from both the original query and the generated medical hypothesis.

3. **Med-KGRM**: Medical Knowledge Graph Retrieval Module  
   Aligns extracted entities to a medical knowledge graph and retrieves multi-hop reasoning paths.

4. **Med-DVM**: Medical Dual-dimensional Validation Module  
   Validates and ranks retrieved KG paths from both semantic and structural dimensions.

5. **Med-LRM**: Medical Language Response Module  
   Builds the final RAG prompt for the downstream large language model.

The main pipeline is:

```text
Input Question
    -> Med-HOM
    -> Med-NM
    -> Med-KGRM
    -> Med-DVM
    -> Med-LRM
    -> RAG Prompt / Answer Generation
```

## Environment

Python 3.9 or 3.10 is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use GPU inference, make sure that the installed PyTorch version matches your CUDA version.

## API Keys

The current implementation reads API keys from environment variables.

For OpenAI-compatible models:

```bash
export OPENAI_API_KEY="your_api_key"
```

For Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

For DashScope / Qwen API:

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

For Windows PowerShell:

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

The repository provides `.env.example` only as a template. The code does not automatically load `.env`, so please set environment variables before running.

## Run

Start the Med-HoDV server:

```bash
python Med-HoDV.py --config ./config/Med_HoDV_example.json
```

By default, the server uses the host and port defined in `config/Med_HoDV_example.json`:

```json
{
  "SERVER_HOST": "0.0.0.0",
  "PORT": 8194
}
```

## Dataset and Knowledge Sources

This repository does not include the original datasets or medical KG resources.  
The following public resources can be used to build or evaluate the system:

- MMCU: https://github.com/Felixgithub2017/MMCU
- MMCU on HuggingFace: https://huggingface.co/datasets/Besteasy/MMCU
- CMB: https://github.com/FreedomIntelligence/CMB
- CMB on HuggingFace: https://huggingface.co/datasets/FreedomIntelligence/CMB
- CMeKG / PyCMeKG: https://pypi.org/project/pycmekg/
- CPubMed-KG: https://cpubmed.openi.org.cn/graphwiki
- DiseaseKG: https://www.selectdataset.com/dataset/ca644287e3134376b1b8dfc04b471bfa

Before running Med-HoDV, the raw KG data should be converted into the local `entity.pkl`, `relation.pkl`, and entity embedding files required by the entity linking module.
