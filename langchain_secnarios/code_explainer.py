from langchain_utils import init_gemini_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

llm = init_gemini_model()


explain_template = """
You are an expert programming tutor.
Explain the following {language} code in simple terms that a beginner can understand.
Describe what each line does step by step.

Code:
{code}
"""

explain_prompt = PromptTemplate(
    template=explain_template,
    input_variables=["language", "code"]
)

code_to_explain = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))
"""

formatted_explain = explain_prompt.format(language="Python", code=code_to_explain)
response1 = llm.invoke([HumanMessage(content=formatted_explain)])

print("=== Code Explanation ===\n")
print(f"Code:\n{code_to_explain}")
print("Explanation:")
print(response1.content)


debug_template = """
You are an expert Python debugger.
The following code has a bug. Find the bug, explain why it is wrong,
and provide the corrected version of the code.

Buggy Code:
{code}

Respond in this format:
Bug Found    : <describe the bug>
Reason       : <why it causes an error>
Fixed Code   :
<corrected code here>
"""

debug_prompt = PromptTemplate(
    template=debug_template,
    input_variables=["code"]
)

buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    average = total / len(numbers)
    return average

result = calculate_average([])
print("Average:", result)
"""

formatted_debug = debug_prompt.format(code=buggy_code)
response2       = llm.invoke([HumanMessage(content=formatted_debug)])

print("\n\n=== Code Debugger ===\n")
print(f"Buggy Code:\n{buggy_code}")
print("Debug Report:")
print(response2.content)



improve_template = """
You are a senior software engineer doing a code review.
Review the following {language} code and suggest:
1. Time complexity (Big O)
2. Space complexity (Big O)
3. At least 2 improvements or best practices

Code:
{code}
"""

improve_prompt = PromptTemplate(
    template=improve_template,
    input_variables=["language", "code"]
)

code_to_review = """
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates

print(find_duplicates([1, 2, 3, 2, 4, 3, 5]))
"""

formatted_improve = improve_prompt.format(language="Python", code=code_to_review)
response3         = llm.invoke([HumanMessage(content=formatted_improve)])

print("\n\n=== Code Review & Improvement ===\n")
print(f"Code:\n{code_to_review}")
print("Review:")
print(response3.content)