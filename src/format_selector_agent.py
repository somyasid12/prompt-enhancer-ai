import os
import re
import json

# Updated imports for AG2/AutoGen v0.4
try:
    from autogen import ConversableAgent
except ImportError:
    try:
        from ag2 import ConversableAgent
    except ImportError:
        print("Neither autogen nor ag2 could be imported. Please install ag2.")
        raise


# LLM Configuration 
llm_config = {
    "config_list": [{
        "model": "x-ai/grok-4-fast:free",
        "temperature": 0.2,
        "api_type": "openai",  # Must stay 'openai'
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    }]
}


# FormatSelectorAgent 

class FormatSelectorAgent:
    """
    Research-based agent to classify user queries and select optimal prompt formats.
    Based on academic research about LLM prompt effectiveness.
    """
    
    def __init__(self):
        self.agent = ConversableAgent(
            name="format_selector_agent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda msg: msg.get("content") and "TERMINATE" in msg["content"],
            code_execution_config={"work_dir": ".", "use_docker": False},
            llm_config=llm_config,
            system_message="""
You are FormatSelectorAgent.
Your task is to analyze the user's raw query and decide which prompt format (JSON, YAML, Markdown, Plain Text) is most effective for this type of task.

=== Decision Logic ===
1. Classify the task into one of:
   - Reasoning/Problem Solving (math, logic, planning, step-by-step)
   - Coding/Programming (code generation, debugging, explanation)
   - Research/Summarization (papers, reports, structured summaries)
   - Open-Ended Chat/Conversation (casual, brainstorming, dialogue)
   - Multimodal Request (image, video, audio generation/editing)

2. Select the best format based on research:
   - JSON → structured output, coding, test cases, evaluation
   - YAML → reasoning chains, multi-step workflows, few-shot examples
   - Markdown → summarization, research papers, structured explanations
   - Plain Text → casual or informal tasks

3. Include a confidence score (0–1) indicating certainty.

=== Output ===
Respond ONLY in JSON with the following schema:
{
  "task_type": "<classification>",
  "chosen_format": "<JSON|YAML|Markdown|Plain Text>",
  "confidence": <0-1>,
  "original_query": "<user's query>"
}
"""
        )
    
    def select_format(self, user_query: str) -> dict:
        """
        Analyze user query and return optimal format selection based on research.
        """
        try:
            # generate_reply method for AG2/AutoGen v0.4
            response = self.agent.generate_reply(
                messages=[{"role": "user", "content": user_query}]
            )
            
            # Text extraction from response
            if isinstance(response, dict):
                response_text = response.get("content", str(response))
            else:
                response_text = str(response)
            
            # Attempt to parse JSON from the response
            try:
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    # Validate the response structure
                    if self._is_valid_response(result):
                        return result
                    else:
                        return self._apply_fallback_classification(user_query)
                else:
                    return self._apply_fallback_classification(user_query)
            except Exception:
                return self._apply_fallback_classification(user_query)
            
        except Exception as e:
            # Complete fallback in case of any errors
            return self._apply_fallback_classification(user_query, error=str(e))
    
    def _is_valid_response(self, result: dict) -> bool:
        """Validate if the response has required fields and valid values."""
        required_keys = ["task_type", "chosen_format", "confidence", "original_query"]
        if not all(key in result for key in required_keys):
            return False
        
        valid_task_types = [
            "Reasoning/Problem Solving", 
            "Coding/Programming", 
            "Research/Summarization",
            "Open-Ended Chat/Conversation", 
            "Multimodal Request"
        ]
        valid_formats = ["JSON", "YAML", "Markdown", "Plain Text"]
        
        if result["task_type"] not in valid_task_types:
            return False
        if result["chosen_format"] not in valid_formats:
            return False
        if not isinstance(result["confidence"], (int, float)) or not (0 <= result["confidence"] <= 1):
            return False
        
        return True
    
    def _apply_fallback_classification(self, user_query: str, error: str = None) -> dict:
        """
        Research-based fallback classification when the LLM fails.
        """
        query_lower = user_query.lower().strip()
        
        # Coding/Programming indicators
        coding_keywords = [
            "function", "code", "script", "program", "debug", "python", "javascript", 
            "java", "c++", "html", "css", "sql", "algorithm", "class", "method", 
            "write code", "programming", "syntax", "api", "database", "framework"
        ]
        
        # Reasoning/Problem Solving indicators
        reasoning_keywords = [
            "solve", "calculate", "math", "logic", "step by step", "analyze", 
            "problem", "plan", "strategy", "optimize", "decision", "compare",
            "evaluate", "reasoning", "proof", "derive"
        ]
        
        # Research/Summarization indicators
        research_keywords = [
            "summarize", "summary", "research", "paper", "report", "analysis",
            "review", "literature", "study", "findings", "evidence", "data",
            "conclusion", "abstract", "overview", "survey"
        ]
        
        # Multimodal indicators
        multimodal_keywords = [
            "image", "picture", "video", "audio", "generate", "create visual",
            "diagram", "chart", "graph", "illustration", "design", "photo"
        ]
        
        # Chat/Conversation indicators (everything else falls here)
        
        # Classification logic with priority order
        if any(keyword in query_lower for keyword in coding_keywords):
            task_type = "Coding/Programming"
            chosen_format = "JSON"
            confidence = 0.8
            
        elif any(keyword in query_lower for keyword in reasoning_keywords):
            task_type = "Reasoning/Problem Solving"
            chosen_format = "YAML"
            confidence = 0.75
            
        elif any(keyword in query_lower for keyword in research_keywords):
            task_type = "Research/Summarization"
            chosen_format = "Markdown"
            confidence = 0.85
            
        elif any(keyword in query_lower for keyword in multimodal_keywords):
            task_type = "Multimodal Request"
            chosen_format = "JSON"
            confidence = 0.7
            
        else:
            # Default to conversational
            task_type = "Open-Ended Chat/Conversation"
            chosen_format = "Plain Text"
            confidence = 0.6
        
        result = {
            "task_type": task_type,
            "chosen_format": chosen_format,
            "confidence": confidence,
            "original_query": user_query
        }
        
        # Displaying error info if any
        if error:
            result["fallback_reason"] = f"LLM error: {error}"
        else:
            result["fallback_reason"] = "Rule-based classification used"
        
        return result