from typing import List

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.anonymizers.azure import AzureAnonymizer
from pii_benchmark.anonymizers.gemini import GeminiAnonymizer
from pii_benchmark.anonymizers.gliner import GlinerAnonymizer
from pii_benchmark.anonymizers.iterative import IterativeAnonymizer
from pii_benchmark.anonymizers.llama import LlamaAnonymizer
from pii_benchmark.anonymizers.llamaclio import LlamaClioAnonymizer
from pii_benchmark.anonymizers.presidio import PresidioAnonymizer
from pii_benchmark.anonymizers.llamarescriber import LlamaRescriberAnonymizer
from pii_benchmark.anonymizers.scrubadub import ScrubadubAnonymizer
from pii_benchmark.anonymizers.textwash import TextWashAnonymizer
from pii_benchmark.anonymizers.uniner import UninerAnonymizer
from pii_benchmark.anonymizers.gpt_anon import GPTAnonymizer
from pii_benchmark.anonymizers.madlib import MadlibAnonymizer
from pii_benchmark.anonymizers.tem import TEMAnonymizer
from pii_benchmark.anonymizers.dp_prompt_gpt import DPPromptAnonymizer

def get_anonymizer(method: str, attributes: List[str]=None, **kwargs) -> Anonymizer:
    match method:
        case "presidio":
            return PresidioAnonymizer()
        case "gliner":
            return GlinerAnonymizer()
        case "gemini":
            return GeminiAnonymizer(
                attributes=[
                    "SSN",
                    "phone number",
                    "credit card number",
                    "email",
                    "name",
                    "address",
                ],
                model_version=kwargs["gemini_version"], 
                prompt_type="anthropic_attributes",
            )
        case "gemini_basic":
            return GeminiAnonymizer(attributes=None, prompt_type="anthropic", model_version=kwargs["gemini_version"])
        case "gemini_full":
            return GeminiAnonymizer(attributes=["SSN", "phone number", "credit card number", "email", "name", "address", \
                                            "race", "citizenship status", "educational attainment", "state of residence", "occupation", \
                                            "marital status", "employment status", "date of birth", "age"], model_version=kwargs["gemini_version"], 
                                    prompt_type="anthropic_attributes")
        case "azure":
            return AzureAnonymizer()
        case "uniner":
            return UninerAnonymizer(attributes=["SSN", "phone number", "credit card number", "email", "name", "address", \
                                                "race", "citizenship status", "educational attainment", "state of residence", "occupation", \
                                                "marital status", "employment status", "date of birth", "age"])
        case "llama":
            return LlamaAnonymizer(attributes=["SSN", "phone number", "credit card number", "email", "name", "address"], model_version=kwargs["llama_version"],
                                            prompt_type="anthropic_attributes")
        case "llama_basic":
            return LlamaAnonymizer(attributes=None, prompt_type="anthropic", model_version=kwargs["llama_version"])
        case "llama_full":
            return LlamaAnonymizer(attributes=["SSN", "phone number", "credit card number", "email", "name", "address", \
                                           "race", "citizenship status", "educational attainment", "state of residence", "occupation", \
                                            "marital status", "employment status", "date of birth", "age"], model_version=kwargs["llama_version"],
                                            prompt_type="anthropic_attributes")
        case "scrubadub":
            return ScrubadubAnonymizer(attributes=attributes)
        case "iterative":
            return IterativeAnonymizer(model_version=kwargs["gpt_version"],
                                       attribute_list=kwargs["attribute_list_iterative"])
        case "textwash":
            return TextWashAnonymizer()
        case "llama_clio":
            return LlamaClioAnonymizer(attributes=None, prompt_type="clio", scenario=kwargs["scenario"])
        case "llama_rescriber":
            return LlamaRescriberAnonymizer(attributes=None, prompt_type="rescriber", scenario=kwargs["scenario"])
        case "gpt":
            return GPTAnonymizer(attributes=["SSN", "phone number", "credit card number", "email", "name", "address"],
                                 prompt_type="anthropic_attributes", model_version=kwargs["gpt_version"])
        case "gpt_basic":
            return GPTAnonymizer(attributes=None, prompt_type="anthropic", model_version=kwargs["gpt_version"])
        case "gpt_full":
            return GPTAnonymizer(attributes=["SSN", "phone number", "credit card number", "email", "name", "address", \
                                            "race", "citizenship status", "educational attainment", "state of residence", "occupation", \
                                            "marital status", "employment status", "date of birth", "age"],
                                 prompt_type="anthropic_attributes", model_version=kwargs["gpt_version"])
        case "gpt_attributes_direct":
            return GPTAnonymizer(prompt_type="anthropic_attributes_direct", attributes=attributes, model_version=kwargs["gpt_version"])
        case "gpt_attributes_all":
            return GPTAnonymizer(prompt_type="anthropic_attributes_all", attributes=None, model_version=kwargs["gpt_version"])
        case "gpt_rescriber":
            return GPTAnonymizer(attributes=None, prompt_type="rescriber", model_version=kwargs["gpt_version"])
        case "gpt_clio":
            return GPTAnonymizer(attributes=None, prompt_type="clio", model_version=kwargs["gpt_version"])
        case "madlib":
            return MadlibAnonymizer(epsilon=kwargs["epsilon"])
        case "tem":
            return TEMAnonymizer(epsilon=kwargs["epsilon"])
        case "dp_prompt_gpt":
            return DPPromptAnonymizer(model_version=kwargs["gpt_version"], temperature=kwargs["temperature"])
        #### new anonymizer here
        
