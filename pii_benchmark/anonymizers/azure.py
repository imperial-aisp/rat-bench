from pii_benchmark.anonymizers.anonymizer import Anonymizer
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

from pii_benchmark.credentials import azure_api_key, azure_resource_link

ENTITIES_TO_REMOVE = [
    "Person",
    "PersonType",
    "Location",
    "Organization",
    "Event",
    "Address",
    "PhoneNumber",
    "Email",
    "URL",
    "IP",
    "DateTime",
    "Quantity",
]

QUANTITY_SUBGROUPS = ["Age", "Currency", "Number"]


class AzureAnonymizer(Anonymizer):
    def __init__(self, attributes=None):
        super().__init__()
        self.recognizer = TextAnalyticsClient(
            endpoint=azure_resource_link,
            credential=AzureKeyCredential(azure_api_key),
        )
        self.entities_to_remove = ENTITIES_TO_REMOVE
        self.quantity_subgroups = QUANTITY_SUBGROUPS
        # if attributes is None or "NRP" in attributes:
        #     self.entities_to_remove = entities_to_remove
        #     self.quantity_subgroups = quantity_subgroups
        # else:
        #     self.entities_to_remove = attributes
        #     self.quantity_subgroups = []

    def anonymize(self, text: str) -> str:

        if len(text)<5000:
            entities_results = self.recognizer.recognize_entities([text])[0]
            if not entities_results.is_error:
                redacted_text = self.remove_entities(text, entities_results.entities)
            else:
                print(f"error: {entities_results.error}")
        else:
            n_chunks = len(text)//5000 + 1
            redacted_text = ""
            for chunk in range(n_chunks):
                entities_results = self.recognizer.recognize_entities([text[chunk*5000:(chunk+1)*5000]])[0]
                if not entities_results.is_error:
                    redacted_text += self.remove_entities(text[chunk*5000:(chunk+1)*5000], entities_results.entities)
                else:
                    print(f"error: {entities_results.error}")
            # entities_results_1 = self.recognizer.recognize_entities([text[:5000]])[0]
            # entities_results_2 = self.recognizer.recognize_entities([text[5000:]])[0]
            # if not entities_results_1.is_error:
            #     redacted_text = self.remove_entities(text[:5000], entities_results_1.entities)
            # else:
            #     print(f"error: {entities_results_1.error}")
            # if not entities_results_2.is_error:
            #     redacted_text += self.remove_entities(text[5000:], entities_results_2.entities)            
            # else:
            #     print(f"error: {entities_results_2.error}")
        # print(redacted_text)
        

        # print(f"{entities_results=}")

        # if not entities_results.is_error:
        #     for entity in entities_results.entities:
        #         if (
        #             (
        #                 (entity.category in self.entities_to_remove)
        #                 or (
        #                     entity.category == "Quantity"
        #                     and entity.subcategory in self.quantity_subgroups
        #                 )
        #             )
        #             and (entity.confidence_score > 0.4)
        #             and (entity.text != "Patient:")
        #             and (entity.text != "Doctor:")
        #         ):
        #             start = entity.offset
        #             end = start + entity.length
        #             redacted_text = (
        #                 redacted_text[:start]
        #                 + ("*" * entity.length)
        #                 + redacted_text[end:]
        #             )

        return redacted_text

    def remove_entities(self, text: str, entities: list) -> str:
        redacted_text = text
        for entity in entities:
            if (entity.category in self.entities_to_remove or (entity.category=="Quantity" and entity.subcategory in self.quantity_subgroups)) \
            and (entity.confidence_score>0.4) and (entity.text!="Patient:") and (entity.text!="Doctor:") and (entity.text!="Patient") and (entity.text!="Doctor"):
                redacted_text = redacted_text.replace(entity.text, "*" * entity.length)
        return redacted_text
