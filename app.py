from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_ibm import WatsonxLLM
import json
import os

app = FastAPI()


class RAGInput(BaseModel):
    chunks: List[str]
    titles: List[str]
    urls: List[str]  # New field
    generated_answer: str


class RelevantChunk(BaseModel):
    chunk: str
    title: str
    url: str  # New field
    relevance: float


class RAGOutput(BaseModel):
    relevant_chunks: List[RelevantChunk]


class TextCorrectionInput(BaseModel):
    text: str


class TextCorrectionOutput(BaseModel):
    corrected_text: str


# WatsonX API details
WATSONX_API_KEY = "E3j_MTus2GwazF6jNwDXL3tS56gb1M-BdfsUT03jbymN"
PROJECT_ID = "806bc272-7655-482a-abba-8da43eaeee92"

# Set environment variable for WatsonX API key
os.environ["WATSONX_APIKEY"] = WATSONX_API_KEY

# Create the LLM
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-1-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 200,
        "min_new_tokens": 0,
        "stop_sequences": [],
        "repetition_penalty": 1,
    },
    project_id=PROJECT_ID,
)


def query_watsonx(prompt):
    try:
        return llm.invoke(prompt)
    except Exception as e:
        raise Exception(f"WatsonX API request failed: {str(e)}")


@app.post("/find_relevant_chunks", response_model=RAGOutput)
async def find_relevant_chunks(rag_input: RAGInput):
    if (
        not rag_input.chunks
        or not rag_input.titles
        or not rag_input.urls
        or not rag_input.generated_answer
    ):
        raise HTTPException(
            status_code=400,
            detail="Chunks, titles, urls, and generated_answer must be provided",
        )

    if len(rag_input.chunks) != len(rag_input.titles) or len(rag_input.chunks) != len(
        rag_input.urls
    ):
        raise HTTPException(
            status_code=400,
            detail="Number of chunks, titles, and urls must be the same",
        )

    relevant_chunks = []
    debug_info = []

    for chunk, title, url in zip(rag_input.chunks, rag_input.titles, rag_input.urls):
        prompt = f"""
        Given the following chunk of text and a generated answer, determine the relevance of the chunk to the answer on a scale of 0 to 1, where 0 is not relevant at all and 1 is highly relevant.

        Chunk:
        {chunk}

        Generated Answer:
        {rag_input.generated_answer}

        Provide only a number between 0 and 1 as the relevance score, with no additional text.
        """

        try:
            watsonx_response = query_watsonx(prompt)
            print(f"WatsonX response for chunk '{title}': {watsonx_response}")

            try:
                relevance = float(watsonx_response.strip())
            except ValueError:
                print(
                    f"Could not convert WatsonX response to float: {watsonx_response}"
                )
                relevance = 0

            debug_info.append(
                {
                    "title": title,
                    "relevance": relevance,
                    "raw_response": watsonx_response,
                }
            )

            if relevance > 0.5:
                relevant_chunks.append(
                    RelevantChunk(
                        chunk=chunk,
                        title=title,
                        url=url,  # Add URL to the RelevantChunk
                        relevance=relevance,
                    )
                )
        except Exception as e:
            print(f"Error processing chunk: {e}")
            debug_info.append({"title": title, "error": str(e)})

    # Sort relevant chunks by relevance score in descending order
    relevant_chunks.sort(key=lambda x: x.relevance, reverse=True)

    print("Debug info:", json.dumps(debug_info, indent=2))

    return RAGOutput(relevant_chunks=relevant_chunks)


@app.post("/correct_text", response_model=TextCorrectionOutput)
async def correct_text(input_data: TextCorrectionInput):
    prompt = f"""
    Correct any mistakes and rephrase the following text. The text may be in French, English, or Dutch.
    Maintain the original meaning and tone, but improve the grammar, spelling, and overall clarity.
    Provide ONLY the corrected and rephrased text as a single sentence, without any additional examples, explanations, or notes.

    Text to correct and rephrase:
    {input_data.text}

    Corrected text:
    """

    try:
        corrected_text = query_watsonx(prompt)

        # Process the response to extract only the corrected text
        lines = corrected_text.split("\n")
        corrected_text = lines[0].strip()  # Take only the first line

        # Remove any remaining explanations or notes
        if ":" in corrected_text:
            corrected_text = corrected_text.split(":", 1)[1].strip()

        return TextCorrectionOutput(corrected_text=corrected_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
