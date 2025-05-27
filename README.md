Updated on 14th August 2024
# RAG Recipes
<p>
📝<a href="https://medium.com/@prasadmahamulkar">Article</a> • Demo & Dataset on: 🤗<a href="https://huggingface.co/prsdm">Hugging Face</a> 
</p>


Large Language Models (LLMs) demonstrate significant capabilities but sometimes generate incorrect but believable responses when they lack information, and this is known as “hallucination.” It means they confidently provide information that may sound accurate but could be incorrect due to outdated knowledge.

Retrieval-Augmented Generation or RAG framework solves this problem by integrating an information retrieval system into the LLM pipeline. Instead of relying on pre-trained knowledge, RAG allows the model to dynamically fetch information from external knowledge sources when generating responses. This dynamic retrieval mechanism ensures that the information provided by the LLM is not only contextually relevant but also accurate and up-to-date.

![diagram](https://github.com/user-attachments/assets/508b3a87-ac46-4bf7-b849-145c5465a6c0)

This repository provides a collection of Jupyter notebooks that demonstrate how to build and experiment with RAG using different frameworks and tools. 

### Details of each notebook:
| Tool                         | LLMs                      | Description                                                        | Notebooks |
|------------------------------|---------------------------|--------------------------------------------------------------------|-----------|
| Weaviate & LangChain       | OpenAI                    | Build a question-answer system focused on providing answers related to the Roman Empire using Weaviate, LangChain, and OpenAI.                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LangChain-Weaviate-OpenAI.ipynb) |
| LangChain & LlamaIndex        | OpenAI                    | Build basic and advanced document RAG workflow using  LangChain, LlamaIndex and OpenAI <a href="https://medium.com/@prasadmahamulkar/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a">article</a>.              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LlamaIndex-LangChain-OpenAI.ipynb) |
| LangChain                   | Mixtral                   | Developed a chatbot that retrieves a summary related to the question from the vector database and generates the answer. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LangChain-Mixtral.ipynb) |
| LangChain                    | llama-2                   | Developed a machine learning expert chatbot (using Q&A dataset) that answers questions related to machine learning only without hallucinating. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LangChain-Llama-2.ipynb) |
