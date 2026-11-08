"""Regressions from real chat failures: fabricated size and flat mixed outfits."""
import json

import pytest

from app.services.llm_conversation import build_conversation_plan, ConversationValidationError


def plan(question="Compose des tenues pour un entretien"):
    return build_conversation_plan({"question": question, "profile": {"wardrobe_size": 8},
        "items": [{"ref": "p1", "category": "robe", "colors": ["rouge"]},
                  {"ref": "p2", "category": "bas", "colors": ["noir"]}]})


def response(text, ideas=None):
    return json.dumps({"response_kind": "advice", "answer": text,
        "ideas": ideas or [], "item_refs": [], "follow_up": ""})


def test_wardrobe_count_is_never_presented_as_clothing_size():
    with pytest.raises(ConversationValidationError, match="taille"):
        plan().render(response("Voici ta tenue. Taille : 8. Style casual."))


def test_flat_dress_and_bottom_list_is_rejected():
    with pytest.raises(ConversationValidationError, match="mélange"):
        plan().render(response("Voici une tenue :\nBas : noir\nHaut : blanc\nRobe : rouge"))


def test_user_provided_size_can_be_referenced():
    text = "Ta taille 38 ne renseigne pas la coupe du vêtement."
    assert json.loads(plan("Je porte une taille 38").render(response(text)))["answer"] == text


def test_dress_and_bottom_as_explicit_alternatives_are_allowed():
    text = "Tu peux choisir la robe, ou partir sur un haut et un bas."
    assert json.loads(plan().render(response(text)))["answer"] == text


def test_generic_bottom_is_not_recast_as_unconfirmed_trousers():
    p=build_conversation_plan({"question":"Compose une tenue", "items":[{"ref":"p1","category":"bas","colors":["noir"]}]})
    result=json.loads(p.render(response("Choisis un pantalon noir. Garde le même pantalon pour une autre option.")))
    assert result["answer"] == "Choisis un bas noir. Garde le même bas pour une autre option."
