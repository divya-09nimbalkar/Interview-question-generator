import json
from pathlib import Path

from interview_question_generator import InterviewQuestionGenerator


def test_load_default_bank() -> None:
    generator = InterviewQuestionGenerator(seed=0)
    assert "Machine Learning" in generator.list_topics()
    assert "Beginner" in generator.list_difficulties()
    assert "Conceptual" in generator.list_types()


def test_generate_filter_topic() -> None:
    generator = InterviewQuestionGenerator(seed=0)
    questions = generator.generate(count=2, topic="NLP")
    assert len(questions) == 2
    assert all(question.topic == "NLP" for question in questions)


def test_suggest_returns_topics_and_difficulties() -> None:
    generator = InterviewQuestionGenerator(seed=0)
    suggestions = generator.suggest("transformer")
    assert "topics" in suggestions
    assert "difficulties" in suggestions
    assert len(suggestions["topics"]) == 3
    assert len(suggestions["difficulties"]) == 3


def test_external_questions_file(tmp_path: Path) -> None:
    custom_questions = [
        {
            "text": "What is the purpose of a confusion matrix?",
            "topic": "Machine Learning",
            "difficulty": "Beginner",
            "qtype": "Conceptual",
            "category": "Technical",
            "tags": ["evaluation", "classification"],
        }
    ]
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps(custom_questions), encoding="utf-8")

    generator = InterviewQuestionGenerator(seed=0, questions_file=questions_file)
    assert generator.list_topics() == ["Machine Learning"]
    results = generator.generate(count=1)
    assert results[0].text == "What is the purpose of a confusion matrix?"
