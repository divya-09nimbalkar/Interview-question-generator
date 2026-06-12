import streamlit as st

from interview_question_generator import InterviewQuestion, InterviewQuestionGenerator


def _get_default_generator() -> InterviewQuestionGenerator:
    return InterviewQuestionGenerator(seed=None, questions_file=None)


def _normalize_filter(value: str) -> str | None:
    return None if value == "(Any)" else value


def _render_filter_summary(filters: dict[str, str | None | int]) -> None:
    with st.expander("Selected filters", expanded=True):
        for label, value in filters.items():
            display_value = value if value not in {None, "(Any)"} else "Any"
            st.write(f"**{label.capitalize()}**: {display_value}")


def _build_generator(use_seed: bool, seed: int) -> InterviewQuestionGenerator:
    if use_seed:
        return InterviewQuestionGenerator(seed=int(seed), questions_file=None)
    return InterviewQuestionGenerator(seed=None, questions_file=None)


def _format_question_prompt(question: InterviewQuestion, index: int, total: int) -> str:
    return (
        f"**Question {index} of {total}**\n\n"
        f"{question.text}\n\n"
        f"*Topic: {question.topic} · Difficulty: {question.difficulty} · "
        f"Type: {question.qtype} · Category: {question.category}*"
    )


def _display_question_cards(
    generator: InterviewQuestionGenerator,
    questions: list[InterviewQuestion],
    include_follow_up: bool,
) -> None:
    for index, question in enumerate(questions, start=1):
        st.markdown(f"### {index}. {question.text}")
        st.markdown(
            f"**Topic:** {question.topic}  \n"
            f"**Difficulty:** {question.difficulty}  \n"
            f"**Type:** {question.qtype}  \n"
            f"**Category:** {question.category}"
        )
        st.markdown("**Tags:** " + ", ".join(question.tags))
        if include_follow_up:
            st.markdown(f"**Follow-up:** {generator._build_follow_up(question)}")

        with st.expander("Show answer", expanded=False):
            st.info(question.answer or "Answer not available for this question.")

        st.markdown("---")


def _generate_with_relaxation(
    generator: InterviewQuestionGenerator,
    count: int,
    topic: str | None,
    difficulty: str | None,
    qtype: str | None,
    category: str | None,
) -> tuple[list[InterviewQuestion], str | None]:
    current_filters = {
        "topic": topic,
        "difficulty": difficulty,
        "qtype": qtype,
        "category": category,
    }

    def try_generate(filters: dict[str, str | None]) -> list[InterviewQuestion]:
        return generator.generate(
            count=count,
            topic=filters["topic"],
            difficulty=filters["difficulty"],
            qtype=filters["qtype"],
            category=filters["category"],
        )

    try:
        questions = try_generate(current_filters)
        if len(questions) == count:
            return questions, None
    except ValueError:
        questions = []

    best_questions = questions
    best_message = None
    fallback_order = ["category", "qtype", "difficulty", "topic"]

    for field in fallback_order:
        if current_filters[field] is None:
            continue
        current_filters[field] = None
        try:
            questions = try_generate(current_filters)
            if len(questions) >= count:
                return (
                    questions,
                    f"Relaxed {field} to fill your request while keeping other filters.",
                )
            if len(questions) > len(best_questions):
                best_questions = questions
                best_message = (
                    f"Relaxed {field} to produce more matching questions."
                    if questions
                    else best_message
                )
        except ValueError:
            continue

    if len(best_questions) >= count:
        return best_questions, best_message
    if best_questions:
        return (
            generator.generate(count=count),
            "Relaxed filters too far to satisfy the requested count, so showing random questions from the full bank.",
        )

    return (
        generator.generate(count=count),
        "No matching questions found. Showing random questions from the full bank.",
    )


def _start_practice_session(
    questions: list[InterviewQuestion],
    include_follow_up: bool,
    generator: InterviewQuestionGenerator,
) -> None:
    st.session_state.practice_questions = questions
    st.session_state.practice_index = 0
    st.session_state.practice_include_follow_up = include_follow_up
    st.session_state.practice_generator = generator
    st.session_state.practice_messages = [
        {
            "role": "assistant",
            "content": _format_question_prompt(questions[0], 1, len(questions)),
        }
    ]
    st.session_state.practice_finished = False


def _render_practice_chat() -> None:
    questions: list[InterviewQuestion] = st.session_state.get("practice_questions", [])
    if not questions:
        st.info("Set filters in the sidebar, then click **Start practice** to begin.")
        return

    total = len(questions)
    index = st.session_state.practice_index
    generator: InterviewQuestionGenerator = st.session_state.practice_generator
    include_follow_up: bool = st.session_state.practice_include_follow_up

    st.caption(f"Practice session · Question {min(index + 1, total)} of {total}")

    for message in st.session_state.practice_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.practice_finished:
        st.success("Session complete. Start a new practice run from the sidebar when you are ready.")
        return

    current = questions[index]
    with st.expander("Reveal model answer", expanded=False):
        st.info(current.answer or "Answer not available for this question.")

    if include_follow_up:
        st.markdown(f"**Follow-up prompt:** {generator._build_follow_up(current)}")

    if prompt := st.chat_input("Type your answer and press Enter…"):
        st.session_state.practice_messages.append({"role": "user", "content": prompt})

        if index >= total - 1:
            st.session_state.practice_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Nice work — you finished this set. "
                        "Adjust filters and click **Start practice** for another round."
                    ),
                }
            )
            st.session_state.practice_finished = True
        else:
            next_index = index + 1
            st.session_state.practice_index = next_index
            next_question = questions[next_index]
            st.session_state.practice_messages.append(
                {
                    "role": "assistant",
                    "content": _format_question_prompt(
                        next_question,
                        next_index + 1,
                        total,
                    ),
                }
            )
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI Interview Question Generator", layout="wide")
    st.title("AI Interview Question Generator")
    st.caption("Portfolio-grade interview question preparation for AI/ML/Data Science.")

    if "practice_questions" not in st.session_state:
        st.session_state.practice_questions = []
    if "practice_messages" not in st.session_state:
        st.session_state.practice_messages = []
    if "practice_index" not in st.session_state:
        st.session_state.practice_index = 0
    if "practice_finished" not in st.session_state:
        st.session_state.practice_finished = False
    if "batch_questions" not in st.session_state:
        st.session_state.batch_questions = []
    if "batch_message" not in st.session_state:
        st.session_state.batch_message = None

    generator = _get_default_generator()

    with st.sidebar:
        st.header("Session settings")

        count = st.number_input("Number of questions", min_value=1, max_value=50, value=5, step=1)

        topic = st.selectbox("Topic", options=["(Any)"] + generator.list_topics(), index=0)
        difficulty = st.selectbox(
            "Difficulty",
            options=["(Any)"] + generator.list_difficulties(),
            index=0,
        )
        qtype = st.selectbox("Question type", options=["(Any)"] + generator.list_types(), index=0)
        category = st.selectbox(
            "Category",
            options=["(Any)"] + generator.list_categories(),
            index=0,
        )

        include_follow_up = st.checkbox("Include follow-up prompts", value=False)

        seed = st.number_input("Random seed (optional)", min_value=0, max_value=2**31 - 1, value=0, step=1)
        use_seed = st.checkbox("Use seed", value=False)

        start_practice_clicked = st.button("Start practice", type="primary", use_container_width=True)
        generate_list_clicked = st.button("Generate list", use_container_width=True)

    tab_practice, tab_browse, tab_suggest = st.tabs(["Practice (chat)", "Browse list", "Suggest"])

    filter_values = {
        "topic": _normalize_filter(topic),
        "difficulty": _normalize_filter(difficulty),
        "qtype": _normalize_filter(qtype),
        "category": _normalize_filter(category),
    }

    if start_practice_clicked or generate_list_clicked:
        try:
            gen = _build_generator(use_seed, int(seed))
            questions, fallback_message = _generate_with_relaxation(
                generator=gen,
                count=int(count),
                topic=filter_values["topic"],
                difficulty=filter_values["difficulty"],
                qtype=filter_values["qtype"],
                category=filter_values["category"],
            )

            if start_practice_clicked:
                _start_practice_session(questions, include_follow_up, gen)
            if generate_list_clicked:
                st.session_state.batch_questions = questions
                st.session_state.batch_generator = gen
                st.session_state.batch_include_follow_up = include_follow_up
                st.session_state.batch_message = fallback_message
                st.session_state.batch_filters = {
                    "topic": topic,
                    "difficulty": difficulty,
                    "type": qtype,
                    "category": category,
                    "count": int(count),
                }
        except Exception as e:
            st.error(str(e))

    with tab_practice:
        _render_practice_chat()

    with tab_browse:
        if st.session_state.batch_questions:
            _render_filter_summary(st.session_state.batch_filters)
            st.subheader("Generated questions")
            if st.session_state.batch_message:
                st.warning(st.session_state.batch_message)
            st.success(f"Generated {len(st.session_state.batch_questions)} question(s)")
            _display_question_cards(
                st.session_state.batch_generator,
                st.session_state.batch_questions,
                st.session_state.batch_include_follow_up,
            )
        else:
            st.info("Set filters in the sidebar and click **Generate list** to view all questions at once.")

    with tab_suggest:
        st.subheader("Suggest topics & difficulty")
        st.caption("Enter a short prompt and the tool suggests the best topic(s) and difficulty level(s).")

        query = st.text_input("Prompt", placeholder="e.g., explain transformers and evaluation metrics")
        top_n = st.slider("Top N", min_value=1, max_value=10, value=3, step=1)

        if st.button("Suggest", key="suggest_btn"):
            if not query.strip():
                st.warning("Enter a prompt.")
            else:
                suggestions = generator.suggest(query=query, top_n=int(top_n))

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Topics")
                    for t, score in suggestions["topics"]:
                        st.write(f"- {t}: {score:.2f}")

                with c2:
                    st.markdown("### Difficulties")
                    for d, score in suggestions["difficulties"]:
                        st.write(f"- {d}: {score:.2f}")


if __name__ == "__main__":
    main()
