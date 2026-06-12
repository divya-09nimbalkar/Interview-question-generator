"""
AI Interview Question Generator
Portfolio-grade interview question preparation tool for AI, ML, and data science.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class InterviewQuestion:
    text: str
    topic: str
    difficulty: str
    qtype: str
    category: str
    tags: Tuple[str, ...]
    answer: str = ""

    def format(self, index: Optional[int] = None, style: str = "plain") -> str:
        prefix = f"{index}. " if index is not None else ""
        tag_string = " ".join(f"[{tag}]" for tag in self.tags)

        if style == "markdown":
            return (
                f"{prefix}**{self.text}**  \n"
                f"- Topic: {self.topic}\n"
                f"- Difficulty: {self.difficulty}\n"
                f"- Type: {self.qtype}\n"
                f"- Category: {self.category}\n"
                f"- Tags: {tag_string}\n"
            )

        return f"{prefix}{self.text} ({self.topic}, {self.difficulty}, {self.qtype})"


class InterviewQuestionGenerator:
    def __init__(self, seed: Optional[int] = None, questions_file: Optional[Path] = None):
        self.seed = seed
        self._questions_file = questions_file
        self._bank = self._load_question_bank(questions_file)
        self._topics = sorted(self._bank["topic"].unique())
        self._difficulties = sorted(self._bank["difficulty"].unique())
        self._types = sorted(self._bank["qtype"].unique())
        self._categories = sorted(self._bank["category"].unique())
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._topic_model = LogisticRegression(max_iter=600)
        self._difficulty_model = LogisticRegression(max_iter=600)
        self._train_inference_models()

    def _load_question_bank(self, questions_file: Optional[Path]) -> pd.DataFrame:
        path = questions_file or Path(__file__).parent / "questions.json"
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".json":
                return self._load_json_question_bank(path)
            if suffix == ".csv":
                return self._load_csv_question_bank(path)
        return self._create_default_question_bank()

    def _load_json_question_bank(self, path: Path) -> pd.DataFrame:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return pd.DataFrame(data)

    def _load_csv_question_bank(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _create_default_question_bank(self) -> pd.DataFrame:
        rows = [
            {
                "text": "Explain the bias-variance tradeoff and how it affects model selection.",
                "topic": "Machine Learning",
                "difficulty": "Beginner",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["regression", "overfitting", "generalization"],
                "answer": "The bias-variance tradeoff balances error from too-simple models (high bias) against error from too-complex models (high variance). Optimal model selection finds a middle ground that minimizes total generalization error.",
            },
            {
                "text": "Describe the architecture of a transformer model and explain why attention is important.",
                "topic": "NLP",
                "difficulty": "Advanced",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["transformers", "attention", "sequence modeling"],
                "answer": "A transformer uses self-attention and feed-forward layers in encoder-decoder stacks. Attention allows the model to weight input tokens based on relevance, capturing long-range dependencies more effectively than recurrence.",
            },
            {
                "text": "What evaluation metrics would you choose for an imbalanced binary classification problem?", 
                "topic": "Machine Learning",
                "difficulty": "Intermediate",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["classification", "metrics", "imbalance"],
                "answer": "For imbalanced data, prefer precision, recall, F1 score, and the ROC AUC rather than accuracy. Precision and recall help measure the tradeoff between false positives and false negatives in the minority class.",
            },
            {
                "text": "How would you approach productionizing an object detection pipeline for video analytics?", 
                "topic": "Computer Vision",
                "difficulty": "Advanced",
                "qtype": "Design",
                "category": "Technical",
                "tags": ["deployment", "real-time", "scalability"],
                "answer": "I would deploy a scalable inference service, use a lightweight detection model, add batching or model quantization, and monitor latency and accuracy in production. Edge processing or streaming architectures can help meet real-time requirements.",
            },
            {
                "text": "Write pseudocode for a training loop that uses early stopping and learning rate decay.",
                "topic": "Deep Learning",
                "difficulty": "Intermediate",
                "qtype": "Coding",
                "category": "Technical",
                "tags": ["optimization", "training", "regularization"],
                "answer": "A training loop checks validation loss each epoch, records the best model, reduces the learning rate on plateau, and stops when no improvement occurs for several epochs. This avoids overfitting while refining the model with smaller updates.",
            },
            {
                "text": "What are the advantages and disadvantages of using pre-trained embeddings for a text classification task?", 
                "topic": "NLP",
                "difficulty": "Intermediate",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["embeddings", "transfer learning", "text"],
                "answer": "Pre-trained embeddings speed up training and transfer useful semantic structure, but they may not capture domain-specific language and can limit fine-tuning. They are often a good starting point for smaller datasets.",
            },
            {
                "text": "How do you explain model drift to a business stakeholder, and what monitoring signals would you implement?", 
                "topic": "MLOps",
                "difficulty": "Intermediate",
                "qtype": "Behavioral",
                "category": "Operational",
                "tags": ["drift", "monitoring", "stakeholder communication"],
                "answer": "Model drift happens when production data distribution changes from training data, causing performance degradation. I would monitor prediction statistics, feature distributions, and key business metrics with alerts for significant shifts.",
            },
            {
                "text": "Describe a time when you had to balance model performance and explainability in a deployed system.",
                "topic": "Behavioral",
                "difficulty": "Beginner",
                "qtype": "Behavioral",
                "category": "Soft Skills",
                "tags": ["explainability", "trade-offs", "decision making"],
                "answer": "I chose a simpler model with acceptable accuracy and used visual explanations like SHAP values to keep stakeholders informed, ensuring the system was both reliable and transparent.",
            },
            {
                "text": "Compare and contrast batch normalization and layer normalization in deep neural networks.",
                "topic": "Deep Learning",
                "difficulty": "Advanced",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["normalization", "training stability", "architectures"],
                "answer": "Batch norm normalizes activations across the batch dimension and works well for CNNs, while layer norm normalizes across features for each sample and is better suited to transformers and RNNs.",
            },
            {
                "text": "How would you design an A/B test to validate a recommendation model before full deployment?", 
                "topic": "Data Science",
                "difficulty": "Intermediate",
                "qtype": "Design",
                "category": "Technical",
                "tags": ["experimentation", "metrics", "causality"],
                "answer": "I would split traffic between the current and new model, define success metrics like click-through rate or revenue lift, run the experiment long enough for statistical power, and control for user segments.",
            },
            {
                "text": "What steps would you take to clean and preprocess a dataset containing missing values, outliers, and categorical features?", 
                "topic": "Data Science",
                "difficulty": "Beginner",
                "qtype": "Technical",
                "category": "Technical",
                "tags": ["preprocessing", "feature engineering", "data quality"],
                "answer": "I would analyze missingness patterns, impute or drop values as appropriate, treat or cap outliers, and encode categorical variables with one-hot or ordinal schemes depending on the model.",
            },
            {
                "text": "Explain the difference between L1 and L2 regularization and when you would use each.",
                "topic": "Machine Learning",
                "difficulty": "Beginner",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["regularization", "sparsity", "generalization"],
                "answer": "L1 regularization encourages sparsity by penalizing absolute weights, useful for feature selection. L2 penalizes squared weights, encouraging smaller weights without forcing zeros, and often improves generalization.",
            },
            {
                "text": "Describe how you would implement cross-validation for time-series forecasting.",
                "topic": "Statistics",
                "difficulty": "Advanced",
                "qtype": "Technical",
                "category": "Technical",
                "tags": ["forecasting", "cross-validation", "temporal data"],
                "answer": "Use time-series split methods that respect temporal order, such as expanding or sliding windows, because random shuffling would leak future information into the training set.",
            },
            {
                "text": "What are the trade-offs between using convolutional neural networks and transformers for image classification?", 
                "topic": "Computer Vision",
                "difficulty": "Advanced",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["CNN", "transformers", "vision"],
                "answer": "CNNs are efficient and rely on locality, while transformers can capture global relationships and scale better with large datasets, but they often require more computation and data.",
            },
            {
                "text": "Write a SQL query to identify the top 5 features correlated with customer churn from a labeled dataset.", 
                "topic": "Data Science",
                "difficulty": "Intermediate",
                "qtype": "Coding",
                "category": "Technical",
                "tags": ["sql", "correlation", "feature selection"],
                "answer": "You can compute correlation coefficients by joining aggregated tables or using analytic functions; then order by absolute correlation and select the top 5 features most associated with churn.",
            },
            {
                "text": "When would you prefer a probabilistic model like Naive Bayes over a discriminative model like logistic regression?", 
                "topic": "Machine Learning",
                "difficulty": "Intermediate",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["probability", "logistic regression", "model selection"],
                "answer": "Use Naive Bayes when features are conditionally independent, data is scarce, or interpretability of class-conditional likelihoods matters; logistic regression is generally better with more data and correlated features.",
            },
            {
                "text": "Describe a time when you simplified a machine learning problem to improve model reliability or performance.", 
                "topic": "Behavioral",
                "difficulty": "Intermediate",
                "qtype": "Behavioral",
                "category": "Soft Skills",
                "tags": ["problem solving", "optimization", "communication"],
                "answer": "I focused on the most predictive features, reduced dimensionality, and chose a simpler model so it was easier to debug and maintain, which improved reliability without sacrificing much performance.",
            },
            {
                "text": "Explain the difference between precision, recall, and F1 score. When is each metric most appropriate?", 
                "topic": "Statistics",
                "difficulty": "Beginner",
                "qtype": "Conceptual",
                "category": "Technical",
                "tags": ["metrics", "classification", "evaluation"],
                "answer": "Precision measures correct positives among predictions, recall measures correct positives among actual positives, and F1 balances both. Use precision when false positives are costly and recall when false negatives are costly.",
            },
            {
                "text": "How would you set up versioning for datasets, model weights, and deployment artifacts in an MLOps pipeline?", 
                "topic": "MLOps",
                "difficulty": "Advanced",
                "qtype": "Design",
                "category": "Operational",
                "tags": ["versioning", "pipelines", "reproducibility"],
                "answer": "I would use dataset versioning tools, store model artifacts with semantic or hash-based versioning, and tie deployments to reproducible pipeline runs to ensure traceability and rollback ability.",
            },
        ]

        return pd.DataFrame(rows)

    def _train_inference_models(self) -> None:
        text_data = self._bank["text"].astype(str)
        X = self._vectorizer.fit_transform(text_data)
        self._topic_model_trained = False
        self._difficulty_model_trained = False

        if len(self._bank["topic"].unique()) > 1:
            self._topic_model.fit(X, self._bank["topic"])
            self._topic_model_trained = True

        if len(self._bank["difficulty"].unique()) > 1:
            self._difficulty_model.fit(X, self._bank["difficulty"])
            self._difficulty_model_trained = True

    def list_topics(self) -> List[str]:
        return self._topics

    def list_difficulties(self) -> List[str]:
        return self._difficulties

    def list_types(self) -> List[str]:
        return self._types

    def list_categories(self) -> List[str]:
        return self._categories

    def generate(
        self,
        count: int = 5,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        qtype: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[InterviewQuestion]:
        df = self._bank
        if topic:
            df = df[df["topic"].str.lower() == topic.lower()]
        if difficulty:
            df = df[df["difficulty"].str.lower() == difficulty.lower()]
        if qtype:
            df = df[df["qtype"].str.lower() == qtype.lower()]
        if category:
            df = df[df["category"].str.lower() == category.lower()]

        if df.empty:
            raise ValueError("No interview questions match the selected filters.")

        sample_size = min(count, len(df))
        sample = df.sample(n=sample_size, random_state=self.seed)
        return [self._record_to_question(record) for record in sample.to_dict(orient="records")]

    def _record_to_question(self, record: dict) -> InterviewQuestion:
        tags = record.get("tags") or ()
        if isinstance(tags, list):
            tags = tuple(tags)
        answer = record.get("answer", "")
        if answer is None or (isinstance(answer, float) and pd.isna(answer)):
            answer = ""
        return InterviewQuestion(
            text=str(record["text"]),
            topic=str(record["topic"]),
            difficulty=str(record["difficulty"]),
            qtype=str(record["qtype"]),
            category=str(record["category"]),
            tags=tags,
            answer=str(answer),
        )

    def suggest(self, query: str, top_n: int = 3) -> dict[str, List[tuple[str, float]]]:
        vector = self._vectorizer.transform([query])

        if self._topic_model_trained:
            topic_probabilities = list(zip(self._topic_model.classes_, self._topic_model.predict_proba(vector)[0]))
        else:
            topic_probabilities = [(topic, 1.0 if i == 0 else 0.0) for i, topic in enumerate(self._topics)]

        if self._difficulty_model_trained:
            difficulty_probabilities = list(zip(self._difficulty_model.classes_, self._difficulty_model.predict_proba(vector)[0]))
        else:
            difficulty_probabilities = [(difficulty, 1.0 if i == 0 else 0.0) for i, difficulty in enumerate(self._difficulties)]

        topic_suggestion = sorted(topic_probabilities, key=lambda row: row[1], reverse=True)[:top_n]
        difficulty_suggestion = sorted(difficulty_probabilities, key=lambda row: row[1], reverse=True)[:top_n]
        return {
            "topics": [(topic, float(score)) for topic, score in topic_suggestion],
            "difficulties": [(difficulty, float(score)) for difficulty, score in difficulty_suggestion],
        }

    def render(self, questions: Sequence[InterviewQuestion], style: str = "plain", include_follow_up: bool = False) -> str:
        lines: List[str] = []
        for index, question in enumerate(questions, start=1):
            lines.append(question.format(index=index, style=style))
            if include_follow_up:
                follow_up = self._build_follow_up(question)
                if style == "markdown":
                    lines.append(f"- Follow-up: {follow_up}")
                else:
                    lines.append(f"    Follow-up: {follow_up}")
        return "\n\n".join(lines)

    def _build_follow_up(self, question: InterviewQuestion) -> str:
        follow_up_templates = {
            "Conceptual": [
                "How would you explain this concept to a non-technical stakeholder?",
                "What are the limitations of this approach in production?",
            ],
            "Coding": [
                "What edge cases should you handle when implementing this solution?",
                "How would you optimize runtime or memory usage?",
            ],
            "Design": [
                "What trade-offs would you consider for scalability and reliability?",
                "How would you measure success once the solution is deployed?",
            ],
            "Behavioral": [
                "What was the most important lesson you learned from that experience?",
                "How did you communicate your findings to stakeholders?",
            ],
            "Technical": [
                "What assumptions are you making when you choose this strategy?",
                "How would you validate your decision with data?",
            ],
        }
        templates = follow_up_templates.get(question.qtype, follow_up_templates["Conceptual"])
        return random.choice(templates)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate curated AI/ML interview questions for technical preparation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--count", "-c", type=int, default=5, help="Number of interview questions to generate.")
    parser.add_argument("--topic", "-t", type=str, help="Filter questions by topic.")
    parser.add_argument("--difficulty", "-d", type=str, choices=["Beginner", "Intermediate", "Advanced"], help="Filter questions by difficulty.")
    parser.add_argument("--type", dest="qtype", type=str, help="Filter questions by question type.")
    parser.add_argument("--category", type=str, help="Filter questions by category.")
    parser.add_argument("--format", choices=["plain", "markdown"], default="plain", help="Output format.")
    parser.add_argument("--output", "-o", type=Path, help="Export generated questions to a file.")
    parser.add_argument("--questions-file", type=Path, help="Load interview questions from a JSON or CSV file.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible question selection.")
    parser.add_argument("--include-follow-up", action="store_true", help="Include follow-up prompts for each question.")
    parser.add_argument("--interactive", action="store_true", help="Ask questions interactively before generating output.")
    parser.add_argument("--quiz", action="store_true", help="Launch an interactive quiz session for generated questions.")
    parser.add_argument("--list-topics", action="store_true", help="List all available topics.")
    parser.add_argument("--list-difficulties", action="store_true", help="List all available difficulty levels.")
    parser.add_argument("--list-types", action="store_true", help="List all available question types.")
    parser.add_argument("--suggest", type=str, help="Suggest the best topic and difficulty for a short prompt.")
    return parser.parse_args()


def prompt_for_generation_options(generator: InterviewQuestionGenerator) -> dict[str, Union[int, Optional[str], bool]]:
    print("Interactive interview question setup")
    print("Press Enter to accept the default or leave a filter blank for randomized selection.")

    count_input = input("How many questions would you like? [5]: ").strip()
    count = int(count_input) if count_input.isdigit() and int(count_input) > 0 else 5

    topic_input = input("Topic (e.g. Machine Learning, NLP, Data Science): ").strip()
    topic = topic_input if topic_input else None

    difficulty_input = input("Difficulty (Beginner, Intermediate, Advanced): ").strip()
    difficulty = difficulty_input if difficulty_input else None

    qtype_input = input("Question type (Conceptual, Coding, Design, Behavioral, Technical): ").strip()
    qtype = qtype_input if qtype_input else None

    category_input = input("Category (e.g. Technical, Operational, Soft Skills): ").strip()
    category = category_input if category_input else None

    follow_up_input = input("Include follow-up prompts? (y/n) [n]: ").strip().lower()
    include_follow_up = follow_up_input in {"y", "yes"}

    if topic and topic not in generator.list_topics():
        available = ", ".join(generator.list_topics())
        print(f"Warning: '{topic}' is not a recognized topic. Available topics: {available}")

    return {
        "count": count,
        "topic": topic,
        "difficulty": difficulty,
        "qtype": qtype,
        "category": category,
        "include_follow_up": include_follow_up,
    }


def ask_quiz_questions(
    questions: Sequence[InterviewQuestion],
    include_follow_up: bool,
    generator: InterviewQuestionGenerator,
) -> None:
    print("\n" + "=" * 72)
    print("Starting quiz. Answer each question and press Enter to continue.")
    print("" + "=" * 72)
    answers = []

    for index, question in enumerate(questions, start=1):
        print(f"\nQuestion {index}/{len(questions)}")
        print("-" * 72)
        print(question.text)
        print(f"(Topic: {question.topic} | Difficulty: {question.difficulty} | Type: {question.qtype})")

        answer = input("Your answer: ").strip()
        follow_up_answer = None

        if include_follow_up:
            follow_up = generator._build_follow_up(question)
            print(f"\nFollow-up prompt: {follow_up}")
            follow_up_answer = input("Your follow-up response: ").strip()

        answers.append({
            "question": question.text,
            "answer": answer,
            "follow_up": follow_up if include_follow_up else None,
            "follow_up_answer": follow_up_answer,
        })

    print("\nQuiz complete! Great work.")
    print(f"You answered {len(answers)} questions.")


def main() -> None:
    args = parse_arguments()
    generator = InterviewQuestionGenerator(seed=args.seed, questions_file=args.questions_file)

    if args.list_topics:
        print("Available topics:")
        print("\n".join(generator.list_topics()))
        return

    if args.list_difficulties:
        print("Available difficulty levels:")
        print("\n".join(generator.list_difficulties()))
        return

    if args.list_types:
        print("Available question types:")
        print("\n".join(generator.list_types()))
        return

    if args.suggest:
        suggestions = generator.suggest(args.suggest)
        print("Suggested topics:")
        for topic, score in suggestions["topics"]:
            print(f"- {topic}: {score:.2f}")
        print("\nSuggested difficulty levels:")
        for difficulty, score in suggestions["difficulties"]:
            print(f"- {difficulty}: {score:.2f}")
        return

    interactive_options = {}
    if args.interactive:
        interactive_options = prompt_for_generation_options(generator)

    questions = generator.generate(
        count=interactive_options.get("count", args.count),
        topic=interactive_options.get("topic", args.topic),
        difficulty=interactive_options.get("difficulty", args.difficulty),
        qtype=interactive_options.get("qtype", args.qtype),
        category=interactive_options.get("category", args.category),
    )

    quiz_follow_up = interactive_options.get("include_follow_up", args.include_follow_up)
    if args.quiz:
        ask_quiz_questions(questions, quiz_follow_up, generator)

    output = generator.render(
        questions,
        style=args.format,
        include_follow_up=quiz_follow_up,
    )
    print(output)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"\nQuestions exported to: {args.output}")


if __name__ == "__main__":
    main()
