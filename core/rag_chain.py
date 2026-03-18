import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


def get_llm():
    """Initialize Gemini model."""
    return ChatGoogleGenerativeAI(
model="gemini-2.5-flash",
        google_api_key="AIzaSyAezLHppaZy7fatp3hUJHOkEZbY_0FBSCc",
        temperature=0.1,
    )


RAG_PROMPT = ChatPromptTemplate.from_template("""
أنت مساعد ذكي متخصص في تحليل المستندات والإجابة على الأسئلة بدقة.

استخدم فقط المعلومات الموجودة في السياق التالي للإجابة على السؤال.
إذا لم تجد الإجابة في السياق، قل بوضوح "لم أجد هذه المعلومات في المستندات المرفوعة."

**السياق من المستندات:**
{context}

**السؤال:** {question}

**تعليمات الإجابة:**
- أجب بشكل واضح ومنظم
- أشر إلى المصدر عند الاقتباس (مثال: [المصدر: اسم_الملف، صفحة X])
- إذا كانت المعلومات من أكثر من مصدر، اذكر كل المصادر
- استخدم نفس لغة السؤال (عربي أو إنجليزي)

**الإجابة:**
""")


def format_context(docs_with_scores: List[tuple]) -> tuple[str, List[Dict]]:
    context_parts = []
    citations = []

    for i, (doc, score) in enumerate(docs_with_scores):
        source = doc.metadata.get("source_name", "مستند غير معروف")
        page = doc.metadata.get("page", "")
        page_str = f"، صفحة {page + 1}" if page != "" else ""

        context_part = f"[{i+1}] المصدر: {source}{page_str}\n{doc.page_content}"
        context_parts.append(context_part)

        citations.append({
            "index": i + 1,
            "source": source,
            "page": page + 1 if page != "" else None,
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "relevance_score": round(1 - score, 3) if score <= 1 else round(1 / (1 + score), 3),
        })

    return "\n\n---\n\n".join(context_parts), citations


def answer_question(question: str, docs_with_scores: List[tuple]) -> Dict[str, Any]:
    if not docs_with_scores:
        return {
            "answer": "❌ لم أجد أي مستندات مرفوعة. الرجاء رفع ملف أولاً.",
            "citations": [],
        }

    context, citations = format_context(docs_with_scores)
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer,
        "citations": citations,
    }
