from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """
Angelina Jolie Voight (Los Angeles, 4 de junho de 1975) é uma atriz, cineasta e ativista humanitária americana. Estreou no cinema ao lado de seu pai, Jon Voight, em Lookin' to Get Out (1982); porém, a carreira dela começou a sério uma década mais tarde, quando participou do filme de baixo orçamento Cyborg 2 (1993), seguido de seu primeiro papel principal em uma grande produção em Hackers (1995). Posteriormente, foi escalada para estrelar os telefilmes biográficos George Wallace (1997), pelo qual ganhou seu primeiro Prêmio Globo de Ouro de Melhor Atriz Coadjuvante em Televisão e recebeu uma indicação ao Prêmio Emmy do Primetime para Melhor Atriz Coadjuvante em minissérie ou telefilme, e Gia (1998), vencendo novamente o Globo de Ouro, só que, desta vez, na categoria de Melhor Atriz em Minissérie ou Filme para Televisão. Em 1999, recebeu elogios por parte dos críticos especializados por sua interpretação como Lisa Rowe no filme Girl, Interrupted, pelo qual ganhou o Oscar de Melhor Atriz Coadjuvante.
"""
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(temperature=0, model="gpt-4")
    llm = ChatOllama(temperature=0, model="gemma3:270m") 
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.content)


if __name__ == "__main__":
    main()
