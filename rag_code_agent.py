import os
import weave

WEAVE_PROJECT = "ai-hacker-cup"
weave_client = weave.init(WEAVE_PROJECT)

# Set the URL for the Mistral model api we'll be using
os.environ["BASE_URL"] = "http://195.242.24.252:8000/v1"

# Select MistralAI models used depending if you want a fast or strong LLM
FAST_LLM = "open-mistral-nemo-2407"
STRONG_LLM = "mistral-large-latest"
os.environ["FAST_LLM"] = FAST_LLM
os.environ["STRONG_LLM"] = STRONG_LLM

# Set the max tokens for the models and how many parallel requests to make in Weave Evaluations
os.environ["MAX_TOKENS"] = "4096"
os.environ["WEAVE_PARALLELISM"] = "1"


from agent import rag_solver, rework_solution
from utils import Problem

practice_dataset_uri = "weave:///parambharat/hackercup/object/practice_dataset:R35fXf9N3FE2IOesg7bRPaPAxiE9YbpirhXO9HcHs8w"
problems_dataset = weave.ref(practice_dataset_uri).get().rows[:]
problems = list(map(lambda x: Problem(**x), problems_dataset))
problem = problems[0]  # Select the first problem

print("Sample Problem:\n\n", problem.model_dump_json(indent=2))

import asyncio
import logging
from nest_asyncio import apply

apply()
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

from agent import REFLECTION_INSTRUCTIONS

print("Loading Retriever")
from retriever import Retriever

retriever = Retriever()

@weave.op
async def rag_solver_with_reflection(
        retriever: Retriever,
        problem: Problem,
        model: str = FAST_LLM,
        temperature: float = 0.7,
        max_iterations: int = 2,
        code_execution_timeout: int = 10,
):
    num_iterations = 0
    test_report = "failed"
    solution = None
    while not test_report == "passed" and num_iterations < max_iterations:
        rag_result = await rag_solver(
            retriever=retriever,
            problem=problem,
            timeout=code_execution_timeout,
            model=model,
            temperature=temperature,
        )
        solution = rag_result["solution"]
        test_report = rag_result["test_report"]
        if test_report == "passed":
            logger.info(f"Passing solution generated successfully for problem: {problem.problem_name}")
            return rag_result
        
        logger.info(f"Solution failed, reworking solution. Problem: {problem.problem_name}")
        rework_result = await rework_solution(
            problem=problem,
            incorrect_solution=solution,
            test_report=test_report,
            model=model,
            temperature=temperature,
            timeout=code_execution_timeout,
        )
        solution = rework_result["solution"]
        test_report = rework_result["test_report"]
        if test_report == "passed":
            logger.info(f"Re-worked solution passed for problem: {problem.problem_name}")
            return {
                "solution": solution,
                "stage": "reflection",
                "test_report": test_report,
            }
        num_iterations += 1
        logger.info(f"Re-worked solution failed, trying iteration {num_iterations}. Problem: {problem.problem_name}")
    logger.info("Failed to generate a solution after {num_iterations} iterations. Problem: {problem.problem_name}")
    return {"solution": solution, "stage": "failed", "test_report": test_report}

async def main():
    reflection_result = await rag_solver_with_reflection(
        retriever, problem, STRONG_LLM, max_iterations=2, code_execution_timeout=30
    )
    return reflection_result

if __name__ == "__main__":
    reflection_result = asyncio.run(main())

    print("*" * 80)
    print(reflection_result["solution"].source_code)
    print("*" * 80)
    print(reflection_result["test_report"])