"""Agentes del orquestador: supervisor, especialistas (investigación y análisis) y sintetizador."""

import warnings

# La consigna sugiere `create_react_agent` (langgraph.prebuilt). En LangGraph 1.x sigue funcionando,
# pero avisa que su sucesor es `langchain.agents.create_agent`; silenciamos ese aviso para que la salida
# de la demo quede limpia.
warnings.filterwarnings("ignore", message="create_react_agent has been moved")
