# TemporalTopicModelling
# Facebook Community Guidelines Topic Analysis

![BERTopic Logo](https://maartengr.github.io/BERTopic/logo.png)

A project analyzing the evolution of Facebook's Community Guidelines using BERTopic for automated topic modeling.

## 📌 Overview

This repository contains:
- A Python script (`Topic_Modeller.py`) for processing historical Facebook policy documents
- Topic modeling pipeline using BERTopic
- Visualization tools for tracking policy changes over time

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/VasuSonii/TemporalTopicModelling.git
   cd facebook-policy-analysis
   pip install -r requirements.txt
python -m spacy download en_core_web_sm
python Topic_Modeller.py --input_dir ./data --output_dir ./results
