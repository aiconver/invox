from datasets import load_dataset
import requests

def load_muc4(n=None):
    return [{
        "text": "On March 5, 1987, a bomb exploded in Bogotá targeting military personnel. The Revolutionary Front claimed responsibility."
    }]


def call_fill_template_api(text, mock=True):
    if mock:
        return {
            "message": "Mocked: Template filled successfully!",
            "filledTemplate": {
                "incident_type": "bombing",
                "perpetrator": "The Revolutionary Front",
                "victim": "military personnel",
                "weapon": "improvised explosive device",
                "location": "Bogotá",
                "date": "March 5, 1987"
            },
            "confidence": 0.93,
            "missingFields": [],
            "warnings": []
        }

    template = {
        "templateName": "muc4-terrorist-incident",
        "fields": {
            "incident_type": {"type": "string", "required": True},
            "perpetrator": {"type": "string", "required": True},
            "victim": {"type": "string"},
            "weapon": {"type": "string"},
            "location": {"type": "string"},
            "date": {"type": "string"}
        }
    }

    res = requests.post("http://localhost:3000/api/fill-template", json={
        "transcript": text,
        "templateDefinition": template
    })

    return res.json() if res.ok else None
