from langchain_core.prompts import ChatPromptTemplate

GENERATE_CONCLUSION_PROMPT = ChatPromptTemplate.from_template("""
You are a professional report conclusion writer.
Your task is to create a clear, concise, and insightful conclusion for the report based on the provided section content.

Guidelines:
- Summarize the key points or findings from the report without repeating content verbatim.
- Connect the conclusion back to the main purpose or objectives of the report.
- Provide closure and leave the reader with a clear takeaway or final insight.
- Keep the tone aligned with the rest of the report.
- Avoid introducing completely new topics not mentioned in the content.
- The length of the conclusion must be 1 paragraph.
- Don't use markdown in the content.
- Don't use placeholder (e.g. for company name)

Sections content:
{content}

Improvement notes (optional):
If `improvements` is provided, integrate them into the new conclusion:
{improvements}
""")

GENERATE_INTRODUCTION_PROMPT = ChatPromptTemplate.from_template("""
You are a professional report introduction and title writer.
Your task is to create a compelling and accurate 'title' and 'introduction' for the report based on the provided section content and conclusion.

Guidelines:
1. Title:
  - Be concise and clear.
  - Accurately reflect the main topic or findings of the report.
  - Avoid unnecessary jargon or overly generic wording.
2. Introduction:
  - Provide clear context and background for the report.
  - State the purpose and scope of the content.
  - Highlight the key themes or insights without revealing all details.
  - Engage the reader and set expectations.
  - The length of the introduction must be no more than 2 paragraphs.
  - Don't repeat content and avoid incomplete sentences.

Extra:
- Don't use markdown in the content.
- Don't use placeholder (e.g. for company name)

Sections content:
{content}
Sections conclusion:
{conclusion}

Improvement notes (optional):
If `improvements` is provided, integrate them into the new introduction:
{improvements}
""")

EVALUATE_CONCLUDER_PROMPT = ChatPromptTemplate.from_template("""
You are a professional report reviewer. Your task is to validate the quality of the introduction, title, and conclusion of the provided report.

Report content:
{content}

Report conclusion:
{conclusion}

Report title and introduction:
Title: {title}
Introduction: {introduction}

Evaluation Criteria (0 to 1 scale):
1. Title:
   - Concise, clear, and accurately represents the report's main topic.
   - Free from jargon, vagueness, or misleading terms.
2. Introduction:
   - Provides clear context or background.
   - States the purpose or objective of the report.
   - Engages the reader and sets expectations.
3. Conclusion:
   - Summarizes the main points from the report.
   - Connects back to the introduction and report purpose.
   - Provides closure or a final takeaway.

Instructions:
- Provide a 'score' between 0 (very poor) and 1 (perfect).
- If applicable, provide a list of 'improvements' to make the conclusion or introduction better.
- If the conclusion and introduction are already optimal, return an empty list for 'improvements'.
""")