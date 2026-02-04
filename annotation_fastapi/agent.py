import re
import subprocess
import os
from litellm import completion
import agent_config
import sys
from dotenv import load_dotenv

########################################################################
# LIST OF PIP LIBS USED FOR THIS 
# mini-swe-agent (?)


########################################################################################
# AGENTIC FUNCTIONS 

load_dotenv('api_keys.env')  # Load environment variables from api_keys.env file

def query_lm(messages: list[dict[str, str]]) -> str:
    response = completion(
        model=agent_config.LLM_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

def parse_action(lm_output: str) -> str:
    """Take LM output, return action"""
    matches = re.findall(
        r"```bash-action\s*\n(.*?)\n```", 
        lm_output, 
        re.DOTALL
    )
    return matches[0].strip() if matches else ""

def execute_action(command: str) -> str:
    """Execute action, return output"""
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        env=os.environ,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return result.stdout


########################################################################################
# PROCESSING 

# just as a draft, read multiline input until /submit is seen
def read_multiline():
    lines = []
    while True:
        line = input()
        if line.strip() == "/submit":
            break
        if "/submit" in line:
            before, _sep, _after = line.partition("/submit")
            before = before.rstrip()
            if before:
                lines.append(before)
            break
        lines.append(line)
    return "\n".join(lines)


########################################################################################
# PROMPTING CYCLE 

# for setting the chat history 
messages = []
default_message = {"role": "system", "content": agent_config.SYSTEM_PROMPT_START}
first_prompt_buffer = None 

# main agentic work loop
def agentic_work_loop(first=False, yolo=False):
    global messages
    global first_prompt_buffer
    print(agent_config.ENTER_TEXT)
    user_message = read_multiline()
    messages.append({"role": "user", "content": user_message})
    if first:
        first_prompt_buffer = messages[-1]
    while True:
        lm_output = query_lm(messages)
        print("LM output", lm_output)
        messages.append({"role": "assistant", "content": lm_output})  # remember what the LM said
        action = parse_action(lm_output)  # separate the action from output
        print("Action", action)
        if action == "exit":
            break

        # ask for confirmation before running the action
        if not yolo:
            confirm = input("\n" + "Proceed to run action? (y/Enter to proceed, n to skip): ").strip().lower()
        else: 
            confirm = 'y'

        if confirm == 'n':
            print("Execution skipped by user.")
            messages.append({"role": "user", "content": "<execution skipped>"})
            continue
        elif confirm == 'y' or confirm == '':
            messages.append({"role": "user", "content": "<accepted>"})
        else:
            # make user input an option in here too
            messages.append({"role": "user", "content": confirm})
            continue

        output = execute_action(action)
        print("Output", output)
        messages.append({"role": "user", "content": output})  # send command output back

# starting point for the first round of annotations 
def start_annotation_request(yolo=False):
    global messages
    messages = []
    messages.append(default_message.copy())
    print(agent_config.HOME_TEXT)
    agentic_work_loop(first=True, yolo=yolo)

# starting point for the iteration rounds of annotations
def iterate_annotation_request(yolo=False):
    global messages
    global first_prompt_buffer
    messages = []
    messages.append(default_message.copy())
    messages.append(first_prompt_buffer.copy())
    messages.append({"role": "system", "content": agent_config.SYSTEM_PROMPT_ITERATE})
    print(agent_config.ITER_TEXT)
    agentic_work_loop(yolo=yolo)


########################################################################################
# Main workflow 

if __name__ == "__main__":
    yolo = False
    start_annotation_request(yolo=yolo)
    while True:
        iterate_annotation_request(yolo=yolo)
