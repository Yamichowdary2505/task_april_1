from langchain_utils import init_gemini_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

llm = init_gemini_model()


single_translation_template = "Translate the following text to {language}:\n\n{text}\n\nRespond with only the translated text. Do not include alternative translations, explanations, or extra content."

single_prompt = PromptTemplate(
    template=single_translation_template,
    input_variables=["language", "text"]
)

text_to_translate = "Artificial Intelligence is transforming the way we live and work."

languages = ["French", "German"]

print("=== Single Language Translations ===\n")
print(f"Original Text : {text_to_translate}\n")

for lang in languages:
    formatted = single_prompt.format(language=lang, text=text_to_translate)
    response  = llm.invoke([HumanMessage(content=formatted)])
    print(f"{lang:10} : {response.content.strip()}")



multi_translation_template = """
Translate the following sentence into the specified languages (French and German only):
{language_list}

Sentence: {text}

Respond in this exact format only, with no alternatives or extra comments:
Language: <translated text>
"""

multi_prompt = PromptTemplate(
    template=multi_translation_template,
    input_variables=["language_list", "text"]
)

sentence       = "Good morning! Have a productive day."
language_list  = "French, German"

formatted_multi = multi_prompt.format(
    language_list=language_list,
    text=sentence
)

response2 = llm.invoke([HumanMessage(content=formatted_multi)])

print("\n\n=== Multi-Language Translation at Once ===\n")
print(f"Original : {sentence}\n")
print(response2.content)


auto_detect_template = """
Detect the language of the following text and translate it to English.

Text: {text}

Respond in this format:
Detected Language: <language>
English Translation: <translation>
"""

auto_detect_prompt = PromptTemplate(
    template=auto_detect_template,
    input_variables=["text"]
)

unknown_texts = [
    "Bonjour, comment ça va?",
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "Hola, ¿cómo estás?",
]

print("\n\n=== Auto-Detect & Translate to English ===\n")
for text in unknown_texts:
    formatted = auto_detect_prompt.format(text=text)
    response3 = llm.invoke([HumanMessage(content=formatted)])
    print(f"Input : {text}")
    print(response3.content.strip())
    print("-" * 50)