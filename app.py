from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os

_ = load_dotenv()
memory = SqliteSaver.from_conn_string(":memory:")

"""
We want to create an essay writing agent which consitst of several steps:
1. Plan: The plan of steps to create an essay.
2. Research: Research, based on the plan, a potential set of documents for the essay.
3. Generate: Based on the plan, generate an essay. After generation, we check if it is good enough or we need a reflection.
4. Reflect: Check if the generated essay is already good enough.
5. Critique and research again: Provide a feedback how the essay can be improved.
"""


class AgentState(TypedDict):
    task: str  # the task we handed over
    plan: str  # the plan the agent created
    draft: str  # the draft version of the essay
    critique: str  # the critique of the current draft
    content: List[str]  # the list of documents to use
    revision_number: int  # the number of revisons we made
    max_revisions: int  # max number of revisons we want. we use it to decide if we want to stop


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# The system prompt for the Planning Agent
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

# The system prompt for the Writer Agent. Writes the content based on the research
WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

# The system prompt for the Reflection Agent. Defines how we critique the essay.
REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

# The system prompt for the Researcher Agent. Defines which documents to search etc.
RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

# The system prompt for the Critique Agent which researches information for the critique. It works on the critique, not the plan
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


# We define that we want to receive a list of strings from the LLM
class Queries(BaseModel):
    queries: List[str]  # we want a list of queries


# Import the tavily client
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# Create the plan node of the graph
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),  # we use the plan prompt as the system message of the plan step
        HumanMessage(content=state['task'])  # the task we hand over, i.e., the topic to write about
    ]
    response = model.invoke(messages)  # We start the model to create a plan
    return {"plan": response.content}  # Add the plan to the agent


# Create the research node.
def research_plan_node(state: AgentState):
    # We restrict it to return response according to the Queries class format.
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    # for all queries the llm returns, we search using tavily for resources
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# Create the generation node
def generation_node(state: AgentState):
    # Get all the resources we collected up to now and create one big string out of it
    content = "\n\n".join(state['content'] or [])
    # Create an input message for the generator agent by using our initial task and the plan we genrated for it
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    # we first create the system message with the writer prompt (with the placeholder replaced by concrete information we collected)
    # we then add the user message with the concrete task
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
    ]
    # call the model with the taks, the plan and the content (RAG like)
    response = model.invoke(messages)
    # We store the response in the draft as the latest version and increase the count
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }


# Create the reflection node
def reflection_node(state: AgentState):
    # Create a critique for the current draft
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}  # write it to the critique attribute of the agent


# Node for the agent to do research for the critique we got
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}  # provides new resources to use for regeneration


# Create the condition if we want to continue
def should_continue(state):
    # simply check if the max number is reached
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# Now create the graph
builder = StateGraph(AgentState)
# the nodes
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
# the start
builder.set_entry_point("planner")
# the conditional edge
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)
# the unconditional edges
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
# Build the graph
graph = builder.compile(checkpointer=memory)
# Visualize it, requires C
from IPython.display import Image

Image(graph.get_graph().draw_png())

# Finally execute it with a sample query
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "what is the difference between langchain and langsmith",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
