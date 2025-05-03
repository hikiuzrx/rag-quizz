from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Correct import for RetrievalQA
import os

# Load Gemini API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

class QuestionRequest(BaseModel):
    type: str
    language: str

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="lang_questions"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Prompt template
prompt = PromptTemplate(
    input_variables=["type", "language", "context"],
    template="Generate {type} questions for learners of {language} based on the following information:\n\n{context}\n\nPlease output questions in clear Q&A format."
)

# QA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    verbose=False,
    chain_type_kwargs={"prompt": prompt}
)

@app.post("/generate-questions")
def generate_questions(req: QuestionRequest):
    try:
        # Fetch relevant documents from the retriever
        docs = retriever.invoke({"query": f"{req.type} {req.language}"})

        # Debugging: Print docs and check its type
        print(f"Docs retrieved: {docs}")
        print(f"Type of docs: {type(docs)}")

        # Check if docs is a list or dict and extract page_content
        if isinstance(docs, list):
            # Extract the page_content from each document in the list
            context = "\n\n".join([doc.get('page_content', 'No content available') for doc in docs])
        elif isinstance(docs, dict):
            # If docs is a single dictionary, get the page_content
            context = docs.get('page_content', 'No content available')
        else:
            # If docs is neither list nor dict, raise an error
            raise ValueError("Unexpected type for docs: neither list nor dict")

        # Debugging: Check the final context
        print(f"Generated context for the model: {context}")

        # Now pass the documents and context correctly to the chain
        result = chain.invoke({
            "input_documents": docs,  # Ensure docs is in the correct format
            "type": req.type,
            "language": req.language,
            "context": context
        })

        # Debugging: Print the result
        print(f"Generated Result: {result}")

        return {"questions": result}

    except Exception as e:
        # Log the error with more context for debugging
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
