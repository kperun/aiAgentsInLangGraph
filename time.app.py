## Time Travel

# Collect the history of previous states
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)
# Get what you want to replay
to_replay = states[-3]

# Replay it by handing over the config to .stream or .invoke
for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)

#----------------------------------
## Go back in time and edit

# Collect the history of previous states
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)
# Get what you want to replay
to_replay = states[-3]

# Change the state
_id = to_replay.values['messages'][-1].tool_calls[0]['id'] # the id is always required to change the right message
to_replay.values['messages'][-1].tool_calls = [{'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in LA, accuweather'},
  'id': _id}]

# Update the state with the right message. Update state 
branch_state = abot.graph.update_state(to_replay.config, to_replay.values)

# Execute it again
for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)

#-------------------------------
## Add message to a state at a given time
# Collect the history of previous states
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)
# Get what you want to replay
to_replay = states[-3]
# Also get again the id
_id = to_replay.values['messages'][-1].tool_calls[0]['id']

# Create a message which you would like to insert. Here we replace the tools response with our own
state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="tavily_search_results_json",
    content="54 degree celcius",
)]}
# Now we add the new message 
branch_and_add = abot.graph.update_state(
    to_replay.config, 
    state_update, 
    as_node="action")

# Finally replay it
for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)

#--------------------------
# Graph example
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.sqlite import SqliteSaver

_ = load_dotenv()

class AgentState(TypedDict):
    """
    lnode: last node
    scratch: a scratch pad for messages
    count: a counter increased on each step
    """
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]

def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }

def should_continue(state):
    # we want to stop once we reach 3 steps
    return state["count"] < 3

builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, # Stop condition
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)

# run it 
thread = {"configurable": {"thread_id": str(1)}}
# Graph invoke starts the graph.
graph.invoke({"count":0, "scratch":"hi"},thread)