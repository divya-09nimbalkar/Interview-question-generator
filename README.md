# AI Interview Question Generator

## Overview
A portfolio-grade Python tool for generating curated AI, ML, and data science interview questions. It supports topic filtering, difficulty selection, follow-up prompts, and export-ready output.

## Features
- Generate technical and behavioral questions across AI/ML topics
- Filter by topic, difficulty, question type, and category
- Export generated questions to a text or Markdown file
- Load questions from an external JSON or CSV question bank
- Suggest relevant topic and difficulty from a short prompt
- Built with `pandas` and `scikit-learn` for structured filtering and inference

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Generate 5 random questions:

```bash
python interview_question_generator.py
```

Generate advanced machine learning questions:

```bash
python interview_question_generator.py --count 5 --topic "Machine Learning" --difficulty Advanced
```

Generate a Markdown study guide and save it:

```bash
python interview_question_generator.py --count 8 --format markdown --output interview_questions.md
```

Use interactive mode to choose options at runtime:

```bash
python interview_question_generator.py --interactive
```

Use quiz mode to answer questions interactively:

```bash
python interview_question_generator.py --quiz
```

Load questions from the external JSON bank:

```bash
python interview_question_generator.py --questions-file questions.json --count 8
```

Suggest the best topic and difficulty for a prompt:

```bash
python interview_question_generator.py --suggest "prepare questions for transformer-based NLP systems"
```

List available options:

```bash
python interview_question_generator.py --list-topics
python interview_question_generator.py --list-difficulties
python interview_question_generator.py --list-types
```

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`

## Author
Divya Nimbalkar
