from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
import json
import os
import random
import re
import sys
from dotenv import load_dotenv
import logging
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
load_dotenv()

JSON_FILE_PATH = os.getenv("JSON_FILE_PATH", "fractions_and_geometry_quiz.json")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Google Generative AI configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Google Generative AI: {e}", exc_info=True)
        print("AI generation features will not work.")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        logging.info("Gemini model initialized.")
    except NameError:
        logging.warning("genai model not initialized. AI generation will not work.")
        model = None
    except Exception as e:
        logging.error(f"Error initializing Gemini model: {e}", exc_info=True)
        model = None
else:
    logging.warning(
        "GOOGLE_API_KEY not found in environment variables. AI generation features will not work."
    )
    model = None


class Question(BaseModel):
    question: str = Field(..., min_length=1)
    options: dict[str, str]
    answer: str = Field(..., min_length=1, max_length=1)
    difficulty: str = Field(..., min_length=1)

    @field_validator("options")
    def validate_options(cls, v):
        if not v or len(v) != 4:
            raise ValueError("Must provide exactly 4 options")
        valid_keys = {"A", "B", "C", "D"}
        if set(v.keys()) != valid_keys:
            raise ValueError("Option keys must be A, B, C, and D")
        for key, value in v.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Option {key} must be a non-empty string")
        return v

    @field_validator("answer")
    def validate_answer(cls, v, info):
        if "options" in info.data and v not in info.data["options"]:
            raise ValueError(
                "Correct answer must be one of the provided options (A, B, C, or D)"
            )
        return v


class UserAnswer(BaseModel):
    question_index: int = Field(..., ge=0)
    user_option: str = Field(..., min_length=1, max_length=1)

    @field_validator("user_option")
    def validate_user_option(cls, v):
        if v.upper() not in {"A", "B", "C", "D"}:
            raise ValueError("User option must be A, B, C, or D")
        return v.upper()


quiz_api = FastAPI()


def generate_single_quiz_text_from_ai(topic: str, difficulty: str):
    """Generates a single quiz question using the AI model."""
    if model is None:
        logging.warning("AI model not available. Cannot generate quiz.")
        return ""

    prompt = f"""
Generate 1 {difficulty} multiple-choice question on the topic "{topic}".
The question must have exactly 4 options (A, B, C, D), and indicate the correct answer.
Also include its difficulty level.
Provide only the question in the specified format below, without any introductory or concluding remarks.

Format:
Question 1: ...
Difficulty: {difficulty}
A) ...
B) ...
C) ...
C) ...
D) ...
Correct Answer: ...
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error during AI content generation: {e}", exc_info=True)
        return ""


def parse_questions_from_text(ai_generated_text: str):
    """Parses raw text output from AI into a list of question dictionaries."""
    questions = []

    first_question_match = re.search(
        r"Question\s*\d*:", ai_generated_text, flags=re.IGNORECASE
    )
    if first_question_match:
        ai_generated_text = ai_generated_text[first_question_match.start() :]
    else:
        return []

    question_blocks = re.split(
        r"Question\s*\d*:\s*", ai_generated_text, flags=re.IGNORECASE
    )

    for block in question_blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if not lines:
            continue

        question_text = ""
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line:
                if not question_text:
                    question_text = line
                else:
                    processed_lines.append(line)

        if not question_text:
            continue

        options = {}
        correct_answer = ""
        difficulty = ""

        for line in processed_lines:
            option_match = re.match(r"^\s*[A-D]\)\s*(.*)", line, flags=re.IGNORECASE)
            difficulty_match = re.match(
                r"^\s*\**Difficulty:\s*\**\s*(.*)", line, flags=re.IGNORECASE
            )
            answer_match = re.match(
                r"^\s*\**Correct Answer:\s*\**\s*([A-D])", line, flags=re.IGNORECASE
            )

            if option_match:
                option_key = line.strip()[0].upper()
                options[option_key] = option_match.group(1).strip()
            elif difficulty_match:
                difficulty = difficulty_match.group(1).strip().lower()
            elif answer_match:
                correct_answer = answer_match.group(1).strip().upper()

        try:
            question_data = {
                "question": question_text,
                "options": options,
                "answer": correct_answer,
                "difficulty": difficulty,
            }
            validated_question = Question(**question_data)
            questions.append(validated_question.model_dump())
        except Exception as e:
            logging.warning(
                f"Skipping question due to parsing/validation error: {e} - Raw block: {block[:100]}...",
                exc_info=True,
            )
            continue

    return questions


def load_all_questions_from_json(json_file_path: str = JSON_FILE_PATH):
    """Loads all quiz questions from a JSON file."""
    all_quiz_questions = []
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            try:
                raw_questions = json.load(f)
                for question_data in raw_questions:
                    try:
                        validated_question = Question(**question_data)
                        all_quiz_questions.append(validated_question.model_dump())
                    except Exception as e:
                        logging.warning(
                            f"Skipping question from JSON due to validation error: {e} - Data: {question_data.get('question', 'N/A')[:50]}...",
                            exc_info=True,
                        )
                        continue
            except json.JSONDecodeError:
                logging.error(
                    f"Error decoding JSON from {json_file_path}", exc_info=True
                )
                all_quiz_questions = []
        logging.info(
            f"Loaded {len(all_quiz_questions)} valid questions from {json_file_path}"
        )
    else:
        logging.warning(
            f"JSON file not found at {JSON_FILE_PATH}. Starting with no pre-loaded questions."
        )
    return all_quiz_questions


def select_questions(
    all_quiz_questions: list, num_questions: int = 10, difficulty: str = None
):
    """Selects a specified number of random questions, optionally filtered by difficulty."""
    difficulty_filtered_questions = all_quiz_questions

    if difficulty:
        difficulty_filtered_questions = [
            q
            for q in all_quiz_questions
            if q.get("difficulty") and q["difficulty"].lower() == difficulty.lower()
        ]
        if not difficulty_filtered_questions:
            logging.warning(f"No questions found for difficulty level: {difficulty}")

    if len(difficulty_filtered_questions) > num_questions:
        return random.sample(difficulty_filtered_questions, num_questions)
    else:
        return difficulty_filtered_questions


loaded_questions = load_all_questions_from_json()


@quiz_api.get("/questions/", response_model=Question)
def get_questions(
    difficulty: str = Query(None, min_length=1),
):
    """Retrieve a single quiz question based on difficulty."""
    logging.info(f"GET /questions/ request received with difficulty={difficulty}")

    # Use a fixed number of 1 to select only one question
    selected_questions = select_questions(
        loaded_questions, num_questions=1, difficulty=difficulty
    )

    if not selected_questions:
        logging.warning(
            f"GET /questions/ failed: No questions found for criteria (difficulty: {difficulty})."
        )
        raise HTTPException(
            status_code=404,
            detail=f"No questions found for the specified criteria (difficulty: {difficulty}).",
        )

    logging.info(f"Successfully retrieved a single question for /questions/.")
    # Return the single question object from the list
    return selected_questions[0]


@quiz_api.post("/answer/")
def check_user_answer(answer: UserAnswer):
    """Check the user's answer against the correct answer."""
    logging.info(
        f"POST /answer/ request received for question_index={answer.question_index}, user_option={answer.user_option}"
    )

    if answer.question_index < 0 or answer.question_index >= len(loaded_questions):
        logging.warning(
            f"POST /answer/ failed: Invalid question index received: {answer.question_index} (out of bounds)."
        )
        raise HTTPException(status_code=400, detail="Invalid question index.")

    correct_question_data = loaded_questions[answer.question_index]
    correct_answer = correct_question_data.get("answer", "").strip().upper()

    if not correct_answer:
        logging.error(
            f"POST /answer/ failed: Could not retrieve correct answer for question index {answer.question_index}."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve correct answer for question index {answer.question_index}.",
        )

    is_correct = answer.user_option == correct_answer
    logging.info(
        f"POST /answer/ result for index {answer.question_index}: user_option={answer.user_option}, correct_answer={correct_answer}, is_correct={is_correct}"
    )

    return {
        "question_index": answer.question_index,
        "user_answer": answer.user_option,
        "is_correct": is_correct,
        "correct_answer": correct_answer,
    }
