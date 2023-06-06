import os
from langchain.chains import RetrievalQA

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


def load_model():
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        #   load_in_8bit=True, # set these options if your GPU supports them!
        #   device_map=1#'auto',
        #   torch_dtype=torch.float16,
        #   low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


class Question(BaseModel):
    question: str


class ResponseData(BaseModel):
    question: str
    answer: str
    sources: list[str]


class Response(BaseModel):
    data: ResponseData


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device_type = os.environ.get("DEVICE_TYPE", "gpu")
if device_type in ["cpu", "CPU"]:
    device = "cpu"
elif device_type in ["mps", "MPS"]:
    device = "mps"
else:
    device = "cuda"

print(f"Running on: {device}")

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": device}
)
# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)
retriever = db.as_retriever()
# Prepare the LLM
# callbacks = [StreamingStdOutCallbackHandler()]
# load the LLM for generating Natural Language responses.
llm = load_model()
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


def get_source_link(doc_path: str):
    """
    Get the source link from the document path

    Args:
        doc_path (str): The path to the document

    Returns:
        str: The source link
    """
    if doc_path.endswith(".txt"):
        with open(doc_path, "r") as f:
            lines = f.readlines()
            last_line = lines[-1]
            if last_line.startswith("Source:"):
                return last_line.split("Source:")[1].strip()
    return doc_path


@app.post("/api/v1/questions")
async def ask_question(question: Question) -> Response:
    query = question.question
    # Get the answer from the chain
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    # # Print the relevant sources used for the answer
    print(
        "----------------------------------SOURCE DOCUMENTS---------------------------"
    )
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    print(
        "----------------------------------SOURCE DOCUMENTS---------------------------"
    )

    sources = [get_source_link(doc.metadata["source"]) for doc in docs]
    sources = list(set(sources))

    return Response(data=ResponseData(question=query, answer=answer, sources=sources))
