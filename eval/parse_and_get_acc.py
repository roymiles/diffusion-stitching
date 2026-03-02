import glob
import io
import json
import multiprocessing
import os
import re
import sys
import traceback
from collections import defaultdict

from math_verify import verify
from parser_helper import (all_boxed_strings, is_equiv, last_boxed_only_string,
                           remove_boxed)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

RE_CHOICE = re.compile(
    r'(?i)\b(?:the\s+)?(?:correct\W*answer\W*is|answer)\b(?:[:\s]|\\n|\\)*'
    r'(?:answer\b(?:[:\s]|\\n|\\)*)?'
    r'(?:\\?boxed\s*\{\s*|\(\s*|\\*)*([A-D])\s*\)?'
)

CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)

_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")

def extract_choice(text: str):
    m = RE_CHOICE.search(text)
    return m.group(1) if m else None

def extract_last_boxed(text: str):
    """
    Returns the content inside the last \\boxed{...} in `text`, handling nested braces.
    Returns None if not found / malformed.
    """
    if not text:
        return None

    matches = list(_BOXED_OPEN_RE.finditer(text))
    for m in reversed(matches):
        i = m.end()  # points just after the opening '{'
        depth = 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            # content is between m.end() and i-1
            return text[m.end(): i - 1].strip()
    return None

def normalize_boxed(s: str):
    if s is None:
        return None
    s = s.strip()

    # Remove surrounding $ ... $
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # Strip trailing punctuation that often sneaks in
    s = s.strip().rstrip(".")
    return s

def extract_python_code_block(text: str) -> str:
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text

def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        parsed_answer = None

        # average forward passes for llada and verifier
        avg_num_fwds = data['avg_num_fwds']
        avg_num_fwds_v = data['avg_num_fwds_v']
        max_avg_num_fwds = data['max_avg_num_fwds']
        wall_time_per_example = data['wall_time'] if "wall_time" in data.keys() else -1

        # raw_generation = raw_generation[0] if isinstance(raw_generation, list) else raw_generation
        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        avg_num_fwds,
        max_avg_num_fwds,
        avg_num_fwds_v,
        wall_time_per_example
    )


def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    # average forward passes for llada and verifier
    avg_num_fwds = data['avg_num_fwds']
    avg_num_fwds_v = data['avg_num_fwds_v']
    max_avg_num_fwds = data['max_avg_num_fwds']
    wall_time_per_example = data['wall_time'] if "wall_time" in data.keys() else -1

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")
        parsed_answers = []

        # 1) collect ALL boxed answers
        try:
            for boxed_expr in all_boxed_strings(raw_generation):
                try:
                    ans = remove_boxed(boxed_expr).strip()
                except Exception:
                    # fallback: pull inside braces if remove_boxed fails
                    m = re.match(r"\\boxed\{(.*)\}\s*$", boxed_expr, re.DOTALL)
                    ans = m.group(1).strip() if m else None

                if ans:
                    parsed_answers.append(ans)
        except Exception:
            parsed_answers = []

        # 2) fallback to <answer>...</answer> if no boxed answers found
        if not parsed_answers:
            # raw_generation = raw_generation[0] if isinstance(raw_generation, list) else raw_generation
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                parsed_answers = [answer_match.group(1).strip()]

        # this is just picking last \boxed now
        if parsed_answers == []:
            chosen_answer = ""
        else:
            chosen_answer = parsed_answers[-1]
            
        parsed_answer = chosen_answer
        is_correct = False
        ok = verify(chosen_answer, ground_truth)
        if not ok:
            ok = is_equiv(chosen_answer, ground_truth)
        is_correct |= ok

        is_correct_str = f"{bcolors.OKGREEN}True{bcolors.ENDC}" if is_correct else f"{bcolors.WARNING}False{bcolors.ENDC}"
        cur_result_str = f"{repr(parsed_answer)} v.s. {repr(ground_truth)} -> {is_correct_str}"
        print(cur_result_str)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        avg_num_fwds,
        max_avg_num_fwds,
        avg_num_fwds_v,
        wall_time_per_example
    )

def _assert_suite_worker(solution_code, setup_code, tests, conn):
    namespace = {"__builtins__": __builtins__}
    sys.stdin = io.StringIO("")
    try:
        if setup_code:
            exec(setup_code, namespace)
        exec(solution_code, namespace)
        for test in tests:
            if isinstance(test, str) and test.strip():
                exec(test, namespace)
        conn.send((True, None))
    except Exception:
        conn.send((False, traceback.format_exc()))
    finally:
        conn.close()


def _run_assert_suite(solution_code, setup_code, tests, timeout=10):
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(
        target=_assert_suite_worker,
        args=(solution_code, setup_code, tests, child_conn),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        parent_conn.close()
        return False, f"Timed out after {timeout} seconds"

    result = None
    if parent_conn.poll():
        result = parent_conn.recv()
    parent_conn.close()

    if result is None:
        return False, "No result returned from evaluation process."
    return result


def parse_mbpp_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    # average forward passes for llada and verifier
    avg_num_fwds = data['avg_num_fwds']
    avg_num_fwds_v = data['avg_num_fwds_v']
    max_avg_num_fwds = data['max_avg_num_fwds']
    wall_time_per_example = data['wall_time'] if "wall_time" in data.keys() else -1

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        raw_generation = item.get("generations", "")
        ground_truth = item.get("ground_truth") or {}

        solution_code = extract_python_code_block(raw_generation)

        base_tests = ground_truth.get("test_list") or []
        base_setup = ground_truth.get("test_setup_code") or ""
        challenge_tests = ground_truth.get("challenge_test_list") or []
        challenge_setup = (
            ground_truth.get("challenge_test_setup_code") or base_setup
        )

        is_correct = False
        error_message = None

        if base_tests:
            base_passed, base_error = _run_assert_suite(
                solution_code, base_setup, base_tests
            )
            if base_passed:
                if challenge_tests:
                    challenge_passed, challenge_error = _run_assert_suite(
                        solution_code, challenge_setup, challenge_tests
                    )
                    is_correct = challenge_passed
                    error_message = challenge_error
                else:
                    is_correct = True
            else:
                error_message = base_error
        else:
            error_message = "No unit tests provided for this example."

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_code": solution_code,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "error": error_message,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        avg_num_fwds,
        max_avg_num_fwds,
        avg_num_fwds_v,
        wall_time_per_example
    )

def parse_countdown_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    # average forward passes for llada and verifier
    avg_num_fwds = data['avg_num_fwds']
    avg_num_fwds_v = data['avg_num_fwds_v']
    max_avg_num_fwds = data['max_avg_num_fwds']
    wall_time_per_example = data['wall_time'] if "wall_time" in data.keys() else -1

    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generations", "")

        # Extract available numbers and target from ground_truth
        numbers = []
        target = None

        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers = ground_truth[0]
            target = ground_truth[1]
        else:
            # Fallback to parsing from question if ground_truth is not in expected format
            numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
            if numbers_match:
                numbers_str = numbers_match.group(1)
                numbers = [int(num.strip()) for num in numbers_str.split(",")]

            target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
            if target_match:
                target = int(target_match.group(1))

        equation = ""
        try:
            equation = remove_boxed(last_boxed_only_string(generated_text))
        except:
            # Try to extract from answer tags
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
            if answer_match:
                equation = answer_match.group(1).strip()
            else:
                equation = generated_text

        # Replace LaTeX operators with Python operators
        equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

        # Check for equation with equals sign and extract only the expression part
        equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if equation_match:
            equation = equation_match.group(1).strip()

        is_correct = False
        result = None

        # Validate and evaluate the equation
        is_valid = validate_equation(equation, numbers)
        if is_valid:
            result = evaluate_equation(equation)
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                total_correct += 1

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        avg_num_fwds,
        max_avg_num_fwds,
        avg_num_fwds_v,
        wall_time_per_example
    )

def parse_humaneval_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    processed_items = []

    # average forward passes for llada and verifier
    avg_num_fwds = data['avg_num_fwds']
    avg_num_fwds_v = data['avg_num_fwds_v']
    max_avg_num_fwds = data['max_avg_num_fwds']
    wall_time_per_example = data['wall_time'] if "wall_time" in data.keys() else -1

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        raw_generation = item.get("generations", "")
        ground_truth = item.get("ground_truth") or {}

        solution_code = extract_python_code_block(raw_generation)

        test_code = ground_truth.get("test", "")
        entry_point = ground_truth.get("entry_point", "")
        
        if entry_point and f"check({entry_point})" not in test_code:
             test_code += f"\ncheck({entry_point})"

        setup_code = "import math\nfrom typing import *"

        is_correct = False
        error_message = None

        if test_code:
            passed, error = _run_assert_suite(
                solution_code, setup_code, [test_code]
            )
            is_correct = passed
            error_message = error
        else:
            error_message = "No tests provided."

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_code": solution_code,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "error": error_message,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        avg_num_fwds,
        max_avg_num_fwds,
        avg_num_fwds_v,
        wall_time_per_example
    )

def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def aggregate_results(directory="."):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))

    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "accuracy": 0.0,
            "questions": [],
        }
    )

    for json_file in json_files:
        filename = os.path.basename(json_file)
        setup_name = extract_setup_name(filename)

        avg_num_fwds     = -1
        max_avg_num_fwds = -1
        avg_num_fwds_v   = -1
        if setup_name:
            print(f"Processing {filename}...")
            if "gsm" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    avg_num_fwds,
                    max_avg_num_fwds,
                    avg_num_fwds_v,
                    wall_time_per_example
                ) = parse_gsm_answers(json_path=json_file)
            elif any(substring in setup_name for substring in ["math"]):
                (
                    correct,
                    processed,
                    detailed_results,
                    avg_num_fwds,
                    max_avg_num_fwds,
                    avg_num_fwds_v,
                    wall_time_per_example
                ) = parse_math_answers(json_path=json_file)
            elif "countdown" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    avg_num_fwds,
                    max_avg_num_fwds,
                    avg_num_fwds_v,
                    wall_time_per_example
                ) = parse_countdown_answers(json_path=json_file)
            elif "mbpp" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    avg_num_fwds,
                    max_avg_num_fwds,
                    avg_num_fwds_v,
                    wall_time_per_example
                ) = parse_mbpp_answers(json_path=json_file)
            elif "humaneval" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    avg_num_fwds,
                    max_avg_num_fwds,
                    avg_num_fwds_v,
                    wall_time_per_example
                ) = parse_humaneval_answers(json_path=json_file)

            setups[setup_name]["correct"] += correct
            setups[setup_name]["processed"] += processed
            setups[setup_name]["questions"].extend(detailed_results)
            setups[setup_name]["avg_num_fwds"] = avg_num_fwds
            setups[setup_name]["max_avg_num_fwds"] = max_avg_num_fwds
            setups[setup_name]["avg_num_fwds_v"] = avg_num_fwds_v
            setups[setup_name]["wall_time_per_example"] = wall_time_per_example

    # Calculate final accuracy and save results
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100 if results["processed"] > 0 else 0
        )

    # Header
    header_format = "{:<40} {:>12} {:>25}" 
    header_line = header_format.format("Setup (task_model_genlen)", "Accuracy", "Avg Steps")
    print(header_line) 
    print("-" * len(header_format))

    # Data rows
    row_format = "{:<40} {:>11.2f}% {:>25.2f}" 
    for setup, results in sorted(setups.items()): 
        avg_steps = results['max_avg_num_fwds'] + results["avg_num_fwds_v"]
        print(row_format.format(setup, results["accuracy"], avg_steps))

    print("=" * len(header_line))

if __name__ == "__main__":
    username = os.environ.get('USER')
    aggregate_results(directory=f"/home/{username}/diffusion-stitching-suppl/out/mbpp_seed15") 
