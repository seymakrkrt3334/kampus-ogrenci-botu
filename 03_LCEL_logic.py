# prompt_template_ornek.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3, groq_api_key=os.getenv("GROQ_API_KEY"))

# Pipeline oluşturmanın LCEL yolu:
# prompt | llm | output_parser
# Her | operatörü bir sonraki adıma geçişi ifade eder

# Karmaşık zincir örneği:
ozet_prompt = ChatPromptTemplate.from_template(
    "Şu metni 3 madde halinde özetle:\n\n{metin}"
)

ceviri_prompt = ChatPromptTemplate.from_template(
    "Şu Türkçe özeti İngilizceye çevir:\n\n{ozet}"
)

parser = StrOutputParser()

# İki zinciri birleştirme
tam_pipeline = (
    {"ozet": ozet_prompt | llm | parser}
    | ceviri_prompt
    | llm
    | parser
)

sonuc = tam_pipeline.invoke({"metin": "LangChain hakkında uzun bir metin..."})

print(sonuc)