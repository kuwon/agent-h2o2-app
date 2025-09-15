
import os
import sys
import argparse

from dotenv import load_dotenv


from agno.agent import Agent, AgentKnowledge
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.markdown import MarkdownKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.embedder.ollama import OllamaEmbedder
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.postgres import PostgresTools


# db_agent = Agent(
#     model=Ollama(
#             id="qwen3:14b",
#             host="localhost:11434"),
#     tools=[PostgresTools(host="localhost", db_name="ai", user="ai", password="ai", port=5432)],
#     markdown=True,
# )
#db_agent.print_response("ai.pension_knowledge라는 테이블의 row count를 알려줘")


def vector_indexing(target_env, model, recreate):
    kb_args = dict()

    doc_path="~/agent-h2o2-app/resources/"
    dev_db_url = os.getenv("DEV_DB_URL")
    prd_db_url = os.getenv("PRD_DB_URL")

    print(f"PRD_DB_URL: {prd_db_url}")
    print(f"DEV_DB_URL: {dev_db_url}")

    db_url = prd_db_url if target_env == "prd" else dev_db_url

    search_agent = None
    if model == "openai":
        vector_db=PgVector(
            table_name="kis_pension_knowledge_openai",
            db_url=db_url,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        )
        kb_args = {
            "vector_db": vector_db
        }
        if recreate:
            kb_args['path'] = doc_path
        openai_knowledge_base_ = MarkdownKnowledgeBase(**kb_args)

        search_agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            knowledge=openai_knowledge_base_,
            #knowledge=agent_knowledge,
            search_knowledge=True,
        )
    elif model == "ollama":

        vector_db=PgVector(
            table_name="kis_pension_knowledge_ollama",
            db_url=db_url,
            embedder=OllamaEmbedder(id="openhermes")
        )

        kb_args = {
            "vector_db": vector_db
        }
        if recreate:
            kb_args['path'] = doc_path
        ollama_knowledge_base_ = MarkdownKnowledgeBase(**kb_args)

        search_agent = Agent(
            model=Ollama(
                    id="qwen3:14b",
                    host="localhost:11434"),
            knowledge=ollama_knowledge_base_,
            #knowledge=agent_knowledge,
            search_knowledge=True,
        )
    else:
        print(f"Invalid model {model}")

    # Run
    if recreate:
        search_agent.knowledge.load(recreate=True, upsert=True)
    search_agent.print_response("연금수령조건을 알려줘", stream=True)

def main():
    load_dotenv()  # 같은 디렉토리의 .env 읽음  

    parser = argparse.ArgumentParser(description="Vector Indexing Script")
    parser.add_argument("model", choices=["ollama", "openai"], help="Select the models for vector indexing")
    parser.add_argument("--recreate", action="store_true", help="Recreate the index instead of just checking")
    args = parser.parse_args()

    # Embed sentence in database
    embeddings = OllamaEmbedder(id="openhermes").get_embedding("연금개시") if args.model == 'ollama' else OpenAIEmbedder(id="text-embedding-3-small").get_embedding("연금개시")

    # Print the embeddings and their dimensions
    print(f"Select {args.model}")
    print(f"Embeddings: {embeddings[:5]}")
    print(f"Dimensions: {len(embeddings)}")

    target_env = "dev"
    vector_indexing(target_env, args.model, args.recreate)


if __name__ == "__main__":
    main()
