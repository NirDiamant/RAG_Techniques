
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import  ServiceContext
from llama_index.core.prompts import PromptTemplate


from llama_index.core.evaluation import BatchEvalRunner


from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator, 
    CorrectnessEvaluator
)
from llama_index.llms.openai import OpenAI



async def evaluate_rag(vector_index, questions, ground_truth_answers):

    """
    Evaluate the RAG model on a set of questions and ground truth answers.

    Args:
    questions: List of questions to evaluate the RAG model on.
    ground_truth_answers: List of ground truth answers for the questions.

    Returns:
    Dictionary containing the evaluation results for faithfulness, relevancy, and correctness.
    """

    gpt4 = OpenAI(temperature=0, model="gpt-4o")
    service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
    faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
    relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)
    correctness_gpt4 = CorrectnessEvaluator(llm=gpt4)

    faithfulness_new_prompt_template = PromptTemplate(""" Please tell if a given piece of information is directly supported by the context.
    You need to answer with either YES or NO.
    Answer YES if any part of the context explicitly supports the information, even if most of the context is unrelated. If the context does not explicitly support the information, answer NO. Some examples are provided below.

    Information: Apple pie is generally double-crusted.
    Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
    Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard, or cheddar cheese.
    It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
    Answer: YES

    Information: Apple pies taste bad.
    Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
    Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard, or cheddar cheese.
    It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
    Answer: NO

    Information: Paris is the capital of France.
    Context: This document describes a day trip in Paris. You will visit famous landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
    Answer: NO

    Information: {query_str}
    Context: {context_str}
    Answer:

    """)

    faithfulness_gpt4.update_prompts({"your_prompt_key": faithfulness_new_prompt_template}) # Update the prompts dictionary with the new prompt template

    runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4, "correctness": correctness_gpt4},
    workers=8,
    )

    eval_results = await runner.aevaluate_queries(
        vector_index.as_query_engine(llm=gpt4), queries=questions, reference=ground_truth_answers   
    )

    return eval_results


    
def get_eval_results(key, eval_results):
    """
    Get the evaluation results for a specific metric.

    Args:
    key: Metric to get the results for (faithfulness, relevancy, correctness).
    eval_results: Dictionary containing the evaluation results for faithfulness, relevancy, and correctness.

    Returns:
    Score for the specified metric.
    """
    results = eval_results[key]
    
    if isinstance(results, float):
        # If the result is already a float (like for "correctness")
        score = results
    else:
        # For other metrics (faithfulness, relevancy) that return a list of results
        correct = sum(1 for result in results if result.passing)
        score = correct / len(results)
    
    print(f"{key.capitalize()} Score: {score:.2f}")
    return score