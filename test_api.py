import uvicorn
from fastapi import FastAPI
app = FastAPI()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from pydantic import BaseModel



class Item(BaseModel):
    query: str
    def __init__(self, query: str) -> None:
        super().__init__(query=query)




embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs = {'device':'cuda'},
                                   encode_kwargs = {'normalize_embeddings':True})




model_name = "llama3.1"
model = Ollama(model=model_name)



client = QdrantClient(path="Database/")
collection_name = "db"
qdrant = Qdrant(client, collection_name, embeddings)



@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.post("/search")
def search(Item:Item):
    query = Item.query
    search_result = qdrant.similarity_search(query=query, k=10)
    i = 0
    list_res = []
    for results in search_result:
        list_res.append({"id":i,
                         "path":results.metadata.get("path"),
                         "content":results.page_content})
        i = i+1
    
    return list_res




@app.post("/chatbot")
async def chatbot(Item:Item):
    query = Item.query
    search_result = qdrant.similarity_search(query=query, k=10)

    context = ""
    mappings = {}

    list_res =[]
    i = 0
    for res in search_result:
        context = context + str(i)+"\n"+res.page_content+"\n\n"
        mappings[i] = res.metadata.get("path")
        list_res.append({"id":i,
                         "path":res.metadata.get("path"),
                         "content":res.page_content})
        i = i +1


    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
                    You are a question answer Bot.
                    Answer user's question using documents given in the context.  
                    Please always reference document id (in square brackets, for example [0],[1]) of the document that was used to make a claim. 
                    Use as many citations and documents as it is necessary to answer question.


                    Reminder:
                    - Response MUST have reference id supporting the document used to make the claim


                <|eot_id|><|start_header_id|>user<|end_header_id|>

                    User Question : "{query}" 

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                
                    Context : "{context}"

                <|start_header_id|>assistant<|end_header_id|> """
    

    template = PromptTemplate(template=prompt, 
                              input_variables=["query","context"])
    
    chain = template | model | StrOutputParser()
    
    response = chain.invoke({"query":query, "context": context})
  

    return {"context":list_res,
            "answer":response}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)







