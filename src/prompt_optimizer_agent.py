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
        "temperature": 0.3,
        "api_type": "openai",  # Must stay 'openai'
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    }]
}


# PromptOptimizerAgent

class PromptOptimizerAgent:
    """
    Research-based agent to optimize prompts using expert-iteration principles.
    Takes FormatSelectorAgent output and creates high-performance prompts.
    """
    
    def __init__(self):
        self.agent = ConversableAgent(
            name="prompt_optimizer_agent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda msg: msg.get("content") and "TERMINATE" in msg["content"],
            code_execution_config={"work_dir": ".", "use_docker": False},
            llm_config=llm_config,
            system_message="""
You are PromptOptimizerAgent.
You receive as input:
- original_query (user's raw query)
- chosen_format (from FormatSelectorAgent)
Your job is to rewrite the query into an optimized, structured prompt that maximizes LLM performance.

=== Core Objectives ===
1. Respect the chosen format (mandatory).
2. Optimize using expert-iteration principles:
   - Clarify ambiguous goals
   - Add reasoning scaffolds (step-by-step instructions if needed)
   - Ensure constraints are explicit (length, tone, style)
   - Eliminate vagueness and bias
3. Suggest the model class best suited for this optimized prompt:
   - Reasoning LLM (multi-step reasoning, planning)
   - Code-specialized LLM (code generation/debugging)
   - Research/long-context LLM (summarization, analysis)
   - General chat LLM (conversation, brainstorming)
   - Multimodal generator (image, video, audio tasks)

=== Process Steps ===
1. Task Understanding: Restate the user's intent; identify domain.
2. Format Integration: Apply the chosen format. Encourage structured reasoning if format is YAML/JSON, or readable sections if Markdown.
3. Enhancement Pass: Refine instructions; insert constraints; add evaluation criteria if needed.
4. Final Output: Provide the optimized prompt and recommended model class.

=== Output Format ===
Always respond ONLY in JSON:

{
  "optimized_prompt": "<final enhanced prompt ready for execution>",
  "model_class": "Reasoning LLM | Code-specialized LLM | Research/long-context LLM | General chat LLM | Multimodal generator"
}

=== Hard Rules ===
- Obey the chosen format from FormatSelectorAgent.
- Never output vendor-specific models (e.g., GPT-4, Claude). Use only model classes.
- Do not remove task-critical details.
- If unsafe or unanswerable, optimize by safe reformulation (avoid outright refusal unless necessary).
"""
        )
    
    def optimize_prompt(self, format_dict: dict) -> dict:
        """
        Takes FormatSelectorAgent output and returns optimized prompt with model recommendation.
        
        Expected input format:
        {
            "task_type": "<classification>",
            "chosen_format": "<JSON|YAML|Markdown|Plain Text>",
            "confidence": <0-1>,
            "original_query": "<user's query>"
        }
        """
        try:
            # Prepare input for the optimizer agent
            input_data = {
                "original_query": format_dict.get("original_query", ""),
                "chosen_format": format_dict.get("chosen_format", "Plain Text"),
                "task_type": format_dict.get("task_type", ""),
                "confidence": format_dict.get("confidence", 0.5)
            }
            
            input_json_str = json.dumps(input_data)
            
            # generate_reply method for AG2/AutoGen v0.4
            response = self.agent.generate_reply(
                messages=[{"role": "user", "content": input_json_str}]
            )
            
            # Text extraction from response
            if isinstance(response, dict):
                response_text = response.get("content", str(response))
            else:
                response_text = str(response)
            
            # Attempt to parse JSON from response
            try:
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    # Validate the response structure
                    if self._is_valid_response(result):
                        return result
                    else:
                        return self._apply_fallback_optimization(format_dict)
                else:
                    return self._apply_fallback_optimization(format_dict)
            except Exception:
                return self._apply_fallback_optimization(format_dict)
            
        except Exception as e:
            # Complete fallback in case of any errors
            return self._apply_fallback_optimization(format_dict, error=str(e))
    
    def _is_valid_response(self, result: dict) -> bool:
        """Validate if the response has required fields and valid values."""
        required_keys = ["optimized_prompt", "model_class"]
        if not all(key in result for key in required_keys):
            return False
        
        valid_model_classes = [
            "Reasoning LLM",
            "Code-specialized LLM", 
            "Research/long-context LLM",
            "General chat LLM",
            "Multimodal generator"
        ]
        
        if result["model_class"] not in valid_model_classes:
            return False
        
        if not result["optimized_prompt"] or not isinstance(result["optimized_prompt"], str):
            return False
        
        return True
    
    def _apply_fallback_optimization(self, format_dict: dict, error: str = None) -> dict:
        """
        Rule-based fallback optimization when the LLM fails.
        """
        original_query = format_dict.get("original_query", "")
        chosen_format = format_dict.get("chosen_format", "Plain Text")
        task_type = format_dict.get("task_type", "")
        
        # Basic optimization based on format and task type
        if chosen_format == "JSON":
            optimized_prompt = f"""Please respond in valid JSON format. 

Task: {original_query}

Requirements:
- Provide structured output
- Include all relevant fields
- Ensure JSON is properly formatted
- Be comprehensive and accurate"""
            
        elif chosen_format == "YAML":
            optimized_prompt = f"""Please respond in YAML format with clear structure.

Task: {original_query}

Requirements:
- Use step-by-step reasoning
- Show your workflow clearly
- Include intermediate steps
- Maintain proper YAML syntax"""
            
        elif chosen_format == "Markdown":
            optimized_prompt = f"""Please respond in well-structured Markdown format.

## Task
{original_query}

## Requirements
- Use clear headings and sections
- Provide comprehensive explanations
- Include examples where relevant
- Maintain readability"""
            
        else:  # Plain Text
            optimized_prompt = f"""Task: {original_query}

Please provide a clear, comprehensive response that:
- Addresses all aspects of the question
- Uses simple, understandable language
- Includes relevant examples or explanations
- Is well-organized and easy to follow"""
        
        # Determine model class based on task type
        if "Coding/Programming" in task_type:
            model_class = "Code-specialized LLM"
        elif "Reasoning/Problem Solving" in task_type:
            model_class = "Reasoning LLM"
        elif "Research/Summarization" in task_type:
            model_class = "Research/long-context LLM"
        elif "Multimodal Request" in task_type:
            model_class = "Multimodal generator"
        else:
            model_class = "General chat LLM"
        
        result = {
            "optimized_prompt": optimized_prompt,
            "model_class": model_class
        }
        
        # Displaying error info if any
        if error:
            result["fallback_reason"] = f"LLM error: {error}"
        else:
            result["fallback_reason"] = "Rule-based optimization used"
        
        return result
