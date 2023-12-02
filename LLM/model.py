from flask import Flask, request, jsonify
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

app = Flask(__name__)

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

@app.route('/get_answer', methods=['POST'])
def get_answer():
    question = request.json.get('question')
    
    if not question:
        return jsonify({'error': 'Question not provided'})
    
    qa_result = qa_bot()
    response = qa_result({'query': question})
    answer = response.get("result")
    sources = response.get("source_documents")

    if not answer:
        return jsonify({'answer': 'Unable to retrieve an answer for the given question'})
    
    # Extracting clean answer
    clean_answer = '\n'.join([line for line in answer.split('\n') if not any(word in line.lower() for word in ['page', 'source'])])

    return jsonify({'answer': clean_answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Change port as needed