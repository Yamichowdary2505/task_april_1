from langchain_utils import init_gemini_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

llm = init_gemini_model()

email_template = """
Write a professional email based on the following details:

To      : {recipient}
Purpose : {purpose}
Tone    : {tone}

The email should have a subject line, greeting, body, and sign-off.
"""

email_prompt = PromptTemplate(
    template=email_template,
    input_variables=["recipient", "purpose", "tone"]
)

recipient  = "HR Manager"
purpose    = "Follow up on a job application I submitted last week for the Python Developer role"
tone       = "Polite and professional"

formatted_email_prompt = email_prompt.format(
    recipient=recipient,
    purpose=purpose,
    tone=tone
)

response = llm.invoke([HumanMessage(content=formatted_email_prompt)])
print("=== Generated Professional Email ===\n")
print(response.content)

cover_letter_template = """
Write a compelling cover letter for a job application with these details:

Applicant Name : {name}
Job Role       : {job_role}
Company        : {company}
Key Skills     : {skills}
Experience     : {experience}

Keep it concise (3 paragraphs), enthusiastic, and tailored to the role.
"""

cover_letter_prompt = PromptTemplate(
    template=cover_letter_template,
    input_variables=["name", "job_role", "company", "skills", "experience"]
)

name       = "Arjun Sharma"
job_role   = "AI Engineer"
company    = "NovaMind Technologies"
skills     = "Python, LangChain, LLMs, Machine Learning"
experience = "1 year of experience building AI-powered applications"

formatted_cover_letter = cover_letter_prompt.format(
    name=name,
    job_role=job_role,
    company=company,
    skills=skills,
    experience=experience
)

response2 = llm.invoke([HumanMessage(content=formatted_cover_letter)])
print("\n=== Generated Cover Letter ===\n")
print(response2.content)