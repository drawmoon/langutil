import os

os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("./.gradio_tmp")


import gradio as gr
import json


DEFAULT_CODE = """from typing import Any
from json_repair import repair_json
from pyiter import it
import requests
import uuid


def parse_flow_output(data: dict[str, Any]) -> dict[str, Any]:
    output = data["outputs"][0]["outputs"][0]
    component_id = output["component_id"]

    if "messages" in output:
        message = it(output["messages"]).first(
            lambda x: x["component_id"] == component_id
        )["message"]

        parsed = repair_json(message, return_objects=True, skip_json_loads=False)
        if isinstance(parsed, dict):
            return parsed
    else:
        ...
    return {"error": "No message found in the output."}


def main(input: str):
    api_key = "sk-Chvon-bk_vf07hr62jjiQM2L_D3D2M2ZkrfRb4YI7j0"
    url = "http://langflow.bdair/api/v1/run/2d5711ed-d5e0-40b3-b5a2-168a7a5c05a9"  # The complete API endpoint URL for this flow

    # Request payload configuration
    payload = {
        "output_type": "chat",
        "input_type": "chat",
        "input_value": input,
    }
    payload["session_id"] = str(uuid.uuid4())

    headers = {"x-api-key": api_key}

    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        # Print response
        data = response.json()
        flow_output = parse_flow_output(data)

        if flow_output:
            if "error" in flow_output:
                return {"error": flow_output["error"]}
            return flow_output
        return {"error": "No valid output received."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Error making API request: {e}"}
    except ValueError as e:
        return {"error": f"Error parsing response: {e}"}
"""


def generate(input: str, code: str) -> str:
    # safe_builtins = {}

    ns = {
        # "__builtins__": safe_builtins,
    }

    try:
        exec(code, ns)
    except Exception:
        return "Failed to execute code. Please check the syntax."

    flow_runner = ns.get("main")
    if flow_runner is None or not callable(flow_runner):
        return "No callable main(input: str) function found in the code."

    try:
        result = flow_runner(input)
    except Exception as e:
        return f"Failed to call main(input: str), error: {e}."

    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except TypeError:
        return "The result returned by main cannot be serialized to JSON."


with gr.Blocks() as demo:
    with gr.Row(equal_height=True):
        textbox = gr.Textbox(
            placeholder="请输入要生成英文的内容...",
            value="江苏2023年分时段电价对居民售电情况的影响分析",
            show_label=False,
            lines=1,
        )
        button = gr.Button("Generate", variant="primary")
    json_preview = gr.Code(label="JSON Preview", language="json", interactive=False)
    code_editor = gr.Code(
        label="Code Editor",
        language="python",
        value=DEFAULT_CODE,
        interactive=True,
        lines=30,
    )

    button.click(
        fn=generate,
        inputs=[textbox, code_editor],
        outputs=json_preview,
    )

if __name__ == "__main__":
    demo.launch(share=True)
