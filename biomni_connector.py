"""
biomni_connector.py
-------------------
Wrapper and connector framework for Biomni A1 agents.

Provides:
  - BiomniAgentWrapper  : wraps a Biomni agent to add execution tracking and async support
  - BiomniAgentConnector: orchestrates multiple wrapped agents in chains or parallel
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    from IPython.display import HTML, Markdown, display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class BiomniAgentWrapper:
    """Wraps a Biomni A1 agent to add execution tracking and async support."""

    def __init__(self, biomni_agent, name: str, description: str = "", color: str = "#3498db"):
        self.biomni_agent = biomni_agent
        self.name = name
        self.description = description
        self.color = color
        self.history: List[Dict[str, Any]] = []
        self.execution_count = 0

    def go(self, prompt: str) -> Any:
        """Execute the Biomni agent and record the interaction."""
        self.execution_count += 1
        start_time = datetime.now()
        result = self.biomni_agent.go(prompt)
        execution_time = (datetime.now() - start_time).total_seconds()

        self.history.append({
            "timestamp": start_time.isoformat(),
            "execution": self.execution_count,
            "prompt": prompt,
            "result": result,
            "execution_time": execution_time,
        })
        return result

    async def go_async(self, prompt: str) -> Any:
        """Async wrapper — runs go() in a thread to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.go, prompt)

    def display_history(self):
        """Print or display execution history as a DataFrame."""
        if not self.history:
            print(f"No history for {self.name}")
            return
        df = pd.DataFrame(self.history)
        df["prompt_length"] = df["prompt"].str.len()
        df["result_length"] = df["result"].astype(str).str.len()
        cols = ["execution", "timestamp", "prompt_length", "result_length", "execution_time"]
        if JUPYTER_AVAILABLE:
            display(HTML(f"<h3 style='color:{self.color};'>📋 {self.name.title()} History</h3>"))
            display(df[cols])
        else:
            print(f"\n=== {self.name.upper()} HISTORY ===")
            print(df[cols].to_string())

    def get_last_result(self) -> Any:
        return self.history[-1]["result"] if self.history else None

    def __repr__(self):
        return f"BiomniAgentWrapper(name={self.name!r}, executions={self.execution_count})"


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class BiomniAgentConnector:
    """Orchestrates multiple Biomni agents as a chain or in parallel."""

    def __init__(self):
        self.agents: Dict[str, BiomniAgentWrapper] = {}
        self.connections: List[Dict[str, Any]] = []
        self.execution_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_agent(
        self,
        biomni_agent,
        name: str,
        description: str = "",
        color: str = "#3498db",
    ) -> BiomniAgentWrapper:
        """Register a Biomni agent with the connector."""
        wrapped = BiomniAgentWrapper(biomni_agent, name, description, color)
        self.agents[name] = wrapped

        if JUPYTER_AVAILABLE:
            display(HTML(f"""
            <div style="background:{color}20;padding:10px;border-left:4px solid {color};margin:5px 0;">
                <strong>🤖 Added: {name}</strong><br>
                <small>{description}</small>
            </div>"""))
        else:
            print(f"✅ Added agent: {name} — {description}")

        return wrapped

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def connect(
        self,
        from_agent: str,
        to_agent: str,
        transform_func: Optional[Callable] = None,
    ):
        """
        Connect two agents. The output of from_agent is optionally transformed
        before being passed as input to to_agent.
        """
        if from_agent not in self.agents or to_agent not in self.agents:
            raise ValueError("Both agents must be registered before connecting.")

        self.connections.append({
            "from": from_agent,
            "to": to_agent,
            "transform": transform_func or (lambda x: str(x)),
        })

        label = f"{from_agent} → {to_agent}" + (" (with transform)" if transform_func else "")
        if JUPYTER_AVAILABLE:
            display(HTML(f"<div style='padding:6px;border:1px solid #dee2e6;margin:3px 0;'>🔗 {label}</div>"))
        else:
            print(f"🔗 Connected: {label}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------

    def visualize_network(self):
        """Display the agent network."""
        if JUPYTER_AVAILABLE:
            html = "<h3>🕸️ Agent Network</h3><div style='font-family:monospace;'>"
            for name, agent in self.agents.items():
                html += f"<div style='color:{agent.color};margin:4px 0;'>🤖 {name} — {agent.description}</div>"
            html += "<br><strong>Connections:</strong><br>"
            for conn in self.connections:
                html += f"<div style='margin-left:20px;'>{conn['from']} → {conn['to']}</div>"
            html += "</div>"
            display(HTML(html))
        else:
            print("\n=== AGENT NETWORK ===")
            for name, agent in self.agents.items():
                print(f"  🤖 {name} — {agent.description}")
            print("Connections:")
            for conn in self.connections:
                print(f"    {conn['from']} → {conn['to']}")

    # ------------------------------------------------------------------
    # Chain execution
    # ------------------------------------------------------------------

    async def execute_chain(
        self,
        start_agent: str,
        initial_prompt: str,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a chain of connected agents starting from start_agent.
        Returns a dict keyed by agent name with step, input, output, and timing.
        """
        if start_agent not in self.agents:
            raise ValueError(f"Starting agent '{start_agent}' not found.")

        results: Dict[str, Any] = {}
        current_output = initial_prompt
        processed: set = set()
        step = 1
        current_name = start_agent

        if JUPYTER_AVAILABLE and show_progress:
            display(HTML(f"""
            <div style="background:#e7f3ff;padding:15px;border:1px solid #b3d9ff;margin:10px 0;">
                <h3>🚀 Agent Chain Starting</h3>
                <strong>Start:</strong> {start_agent}<br>
                <strong>Prompt:</strong> {str(initial_prompt)[:120]}{'…' if len(str(initial_prompt)) > 120 else ''}
            </div>"""))

        while current_name and current_name not in processed:
            agent = self.agents[current_name]

            if JUPYTER_AVAILABLE and show_progress:
                display(HTML(f"""
                <div style="background:{agent.color}20;padding:10px;border-left:4px solid {agent.color};margin:8px 0;">
                    <h4>Step {step}: {current_name} 🤖</h4>
                    <em>Running…</em>
                </div>"""))

            output = await agent.go_async(current_output)
            exec_time = agent.history[-1]["execution_time"] if agent.history else 0

            results[current_name] = {
                "step": step,
                "input": current_output,
                "output": output,
                "agent": agent,
                "execution_time": exec_time,
            }

            if JUPYTER_AVAILABLE and show_progress:
                display(HTML(f"""
                <div style="background:{agent.color}20;padding:10px;border-left:4px solid {agent.color};margin:8px 0;">
                    <h4>Step {step}: {current_name} ✅ ({exec_time:.1f}s)</h4>
                    <strong>Output:</strong> {str(output)[:160]}{'…' if len(str(output)) > 160 else ''}
                </div>"""))

            processed.add(current_name)
            step += 1

            # Find next agent
            next_name = None
            for conn in self.connections:
                if conn["from"] == current_name:
                    current_output = conn["transform"](output)
                    next_name = conn["to"]
                    break
            current_name = next_name

        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "chain",
            "start_agent": start_agent,
            "steps": len(results),
            "total_time": sum(r["execution_time"] for r in results.values()),
        })

        return results

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    async def execute_parallel(
        self,
        agent_prompts: Dict[str, str],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute multiple agents concurrently, each with its own prompt.
        Returns a dict keyed by agent name with result and timing.
        """
        if JUPYTER_AVAILABLE and show_progress:
            display(HTML(f"""
            <div style="background:#fff3cd;padding:15px;border:1px solid #ffeaa7;margin:10px 0;">
                <h3>⚡ Parallel Execution</h3>
                <strong>Agents:</strong> {', '.join(agent_prompts)}
            </div>"""))

        tasks = [
            (name, self.agents[name].go_async(prompt))
            for name, prompt in agent_prompts.items()
            if name in self.agents
        ]

        results: Dict[str, Any] = {}
        for name, task in tasks:
            result = await task
            agent = self.agents[name]
            exec_time = agent.history[-1]["execution_time"] if agent.history else 0
            results[name] = {"result": result, "execution_time": exec_time}

            if JUPYTER_AVAILABLE and show_progress:
                display(HTML(f"""
                <div style="background:{agent.color}20;padding:8px;border-left:3px solid {agent.color};margin:4px 0;">
                    <strong>{name} ✅</strong> ({exec_time:.1f}s):<br>
                    {str(result)[:120]}{'…' if len(str(result)) > 120 else ''}
                </div>"""))

        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "parallel",
            "agents": list(agent_prompts),
            "total_time": sum(r["execution_time"] for r in results.values()),
        })

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def create_results_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert chain results to a tidy DataFrame."""
        rows = []
        for name, r in results.items():
            rows.append({
                "step": r["step"],
                "agent": name,
                "input_chars": len(str(r["input"])),
                "output_chars": len(str(r["output"])),
                "execution_time_s": r["execution_time"],
                "input_preview": str(r["input"])[:60],
                "output_preview": str(r["output"])[:60],
            })
        return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    def display_execution_summary(self):
        if not self.execution_log:
            print("No executions yet.")
            return
        df = pd.DataFrame(self.execution_log)
        if JUPYTER_AVAILABLE:
            display(HTML("<h3>📊 Execution Summary</h3>"))
            display(df)
        else:
            print(df.to_string())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_solution(output: str) -> str:
    """Extract content between <solution>…</solution> tags, or return raw output."""
    match = re.search(r"<solution>(.*?)</solution>", str(output), re.DOTALL)
    return match.group(1).strip() if match else str(output)
