This repository provides a collection of Jupyter notebooks that demonstrate how to build and experiment with RAG using different frameworks and tools (updated on 14/08/2024).
# RAG-Recipes
<p>
  🤗<a href="https://huggingface.co/prsdm">Hugging Face</a> • 📝<a href="https://medium.com/@prasadmahamulkar">Articles</a>
</p>

Large Language Models (LLMs) demonstrate significant capabilities but sometimes generate incorrect but believable responses when they lack information, and this is known as “hallucination.” It means they confidently provide information that may sound accurate but could be incorrect due to outdated knowledge.

Retrieval-Augmented Generation or RAG framework solves this problem by integrating an information retrieval system into the LLM pipeline. Instead of relying on pre-trained knowledge, RAG allows the model to dynamically fetch information from external knowledge sources when generating responses. This dynamic retrieval mechanism ensures that the information provided by the LLM is not only contextually relevant but also accurate and up-to-date.
### Architecture diagram
![diagram](https://github.com/user-attachments/assets/2be3f7f0-fcd3-4d7a-8231-eb8efcb693e7)

### LangChain

LangChain is a framework for building applications with LLMs. It provides abstractions and utilities for creating robust AI applications, such as chatbots, question-answering systems, and knowledge bases. LangChain offers customization options for adjusting the retrieval procedure to suit specific requirements. It generates multiple parallel queries to cover different aspects of the original query and retrieves relevant documents from a vector store.

### LlamaIndex

LlamaIndex is a framework for building applications using large language models (LLMs). It provides tools for ingesting, managing, and querying data, allowing you to create "chat with your data" experiences. LlamaIndex integrates with vector databases like Weaviate to enable retrieval-augmented generation (RAG) systems, where the LLM is combined with an external storage provider to access specific facts and contextually relevant information.

### Weaviate

Weaviate is a vector database that allows you to store and query data using semantic search. It provides a scalable and efficient way to manage large amounts of unstructured data, such as text, images, and audio. Weaviate uses machine learning models to encode data into high-dimensional vectors, enabling fast and accurate retrieval of relevant information based on semantic similarity.

### Detailed explanation of each notebook:
| Tool                         | LLMs                      | Description                                                        | Notebooks |
|------------------------------|---------------------------|--------------------------------------------------------------------|-----------|
| Weaviate & LangChain       | OpenAI                    | Build a question-answer system focused on providing answers related to the Roman Empire using Weaviate, LangChain, and OpenAI.                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LangChain-Weaviate-openai.ipynb) |
| LangChain & LlamaIndex        | OpenAI                    | Build basic and advanced document RAG workflow using  LangChain, LlamaIndex and OpenAI <a href="https://medium.com/@prasadmahamulkar/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a">article</a>.              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LlamaIndex-LangChain-openai.ipynb) |
| LangChain                   | Mixtral                   | Developed a chatbot that retrieves a summary related to the question from the vector database and generates the answer. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LangChain-Mixtral.ipynb) |
| LangChain                    | llama-2                   | Developed a machine learning expert chatbot (using Q&A dataset) that answers questions related to machine learning only without hallucinating. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prsdm/rag-recipes/blob/main/LangChain-Llama-2.ipynb) |
