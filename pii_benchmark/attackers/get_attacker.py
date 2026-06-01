from pii_benchmark.attackers.deepseek import DeepSeekAttacker
from pii_benchmark.attackers.gemini import GeminiAttacker
from pii_benchmark.attackers.gpt import GPTAttacker
from pii_benchmark.attackers.llama import LlamaAttacker


def get_attacker(attacker, model_version):
    match attacker:
        case "deepseek":
            print("DeepSeek attacker")
            return DeepSeekAttacker(model_version)
        case "gemini":
            print("Gemini attacker")
            return GeminiAttacker(model_version)
        case "llama":
            print("Llama attacker")
            return LlamaAttacker(model_version)
        case "gpt":
            print("GPT attacker")
            return GPTAttacker(model_version)
        case _:
            raise ValueError(f"Unknown attacker: {attacker!r}. Valid options: deepseek, gemini, llama, gpt")