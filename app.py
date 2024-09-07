import re
import streamlit as st
from requests import request
import json



st.title("Retrival Augmented Generation")
question = st.text_input("Your Question")

if st.button("Ask"):

    url = "http://127.0.0.1:8002/chatbot"

    payload = json.dumps({"query": question})
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    response = request("POST", url, headers=headers, data=payload)

    try:

        response_json = response.json()
        answer = response_json["answer"]
        documents = response_json['context']

    except json.JSONDecodeError:

        st.error("Invalid JSON response from the server.")
        answer = ""
        documents = []
    
    if answer:
        rege = re.compile("\[Document\ [0-9]+\]|\[[0-9]+\]")
        m = rege.findall(answer)
        num = []
        for n in m:
            num = num + [int(s) for s in re.findall(r'\b\d+\b', n)]


        st.markdown(answer)
        documents = json.loads(response.text)['context']
        show_docs = []
        for n in num:
            for doc in documents:
                if int(doc['id']) == n:
                    show_docs.append(doc)
        
        
        for doc in show_docs:
            with st.expander(str(doc['id'])+" - "+doc['path']):
                st.write(doc['content'])
                with open(doc['path'], 'rb') as f:
                    st.download_button("Download file", f, file_name=doc['path'].split('/')[-1])