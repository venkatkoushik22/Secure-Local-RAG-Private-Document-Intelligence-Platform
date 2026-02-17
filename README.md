# RAG using Llama3, Langchain and ChromaDB
RAG using Llama3, Langchain and ChromaDB

## Objective 🎯

This project utilizes Llama3 Langchain and ChromaDB to establish a Retrieval Augmented Generation (RAG) system. This system empowers you to ask questions about your documents, even if the information wasn't included in the training data for the Large Language Model (LLM). Retrieval Augmented Generation works by first performing a retrieval step when presented with a question. This step fetches relevant documents from a special vector database, where the documents have been indexed.

### Definitions 📝

* **LLM:** Large Language Model
* **Llama3:** LLM developed by Meta
* **Langchain:** Framework designed to streamline the creation of applications utilizing LLMs
* **Vector database:** Database that organizes data using high-dimensional vectors
* **ChromaDB:** Vector database
* **RAG:** Retrieval Augmented Generation (see below for more details)

### Model Details 

* **Model:** Llama 3
* **Variation:** 8b-chat-hf (8b: 8 Billion parameters; hf: HuggingFace)
* **Version:** V1
* **Framework:** Transformers

The pre-trained Llama3 model is fine-tuned with over 15 Trillion tokens and boasts 8 to 70 Billion parameters, making it one of the most powerful open-source models available. It offers significant advancements over the previous Llama2 model.


## Conclusions 

This project successfully implemented a Retrieval Augmented Generation (RAG) solution by leveraging Langchain, ChromaDB, and Llama3 as the LLM. To evaluate the system's performance, we utilized the EU AI Act from 2023. The results demonstrated that the RAG model delivers accurate answers to questions posed about the Act.

**Future Work** 

To further enhance the solution, we will focus on refining the RAG implementation. This will involve optimizing the document embeddings and exploring the use of more intricate RAG architectures.

---
**META LLAMA3 GENAI Real World UseCases End To End Implementation Guides**

1. Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/fsdp-qlora-distributed-llama3.ipynb)

2. Deploy Llama 3 on Amazon SageMaker : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/deploy-llama3.ipynb)

3. RAG using Llama3, Langchain and ChromaDB : [👉Implementation Guide 1▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/rag-using-llama3-langchain-and-chromadb.ipynb)

4. Prompting Llama 3 like a Pro : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/prompting-llama-3-like-a-pro.ipynb)

5. Test Llama3 with some Math Questions : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/test-llama3-with-some-math-questions.ipynb)

6. Llama3 please write code for me : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/llama3-please-write-code-for-me.ipynb)

7. Run LLAMA-3 70B LLM with NVIDIA endpoints on Amazing Streamlit UI : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/LLAMA3-70B-LLM-with-NVIDIA)

8. Llama 3 ORPO Fine Tuning : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Llama-3-ORPO-Fine-Tuning)

9. Meta's LLaMA3-Quantization : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/LLaMA3-Quantization)

10. Finetune Llama3 using QLoRA : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/finetune-llama3-using-qlora.ipynb)

11. Llama3 Qlora Inference : [👉Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/llama3-qlora-inference.ipynb)

12. Beam_Llama3-8B-finetune_task : [Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/Beam_Llama3-8B-finetune_task.py)

13. Llama-3 Finetuning on custom dataset with Unsloth : [Implementation Guide▶️](https://github.com/GURPREETKAURJETHRA/Meta-LLAMA3-GenAI-UseCases-End-To-End-Implementation-Guides/blob/main/GENAI_NOTEBOOKS/Llama-3_Finetuning_on_custom_dataset_with_unsloth.ipynb)
