import re
from collections import defaultdict


def normalize_name(value):
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_name(value):
    text = normalize_name(value)
    noise_tokens = {
        "gm", "g", "kg", "ml", "ltr", "liter", "litre",
        "pcs", "pc", "piece", "pieces", "pack", "pkt",
        "box", "bag", "btl", "can", "tin", "jar",
        "full", "half", "small", "medium", "large",
        "special", "regular", "classic", "premium",
        "new", "deshi", "china", "big", "all",
        "the", "and", "with", "without"
    }
    tokens = []
    for token in text.split():
        if token in noise_tokens or token.isdigit() or len(token) <= 1:
            continue
        tokens.append(token)
    return set(tokens)


def infer_product_intent(name):
    tokens = tokenize_name(name)

    beef_tokens = {"beef", "cow", "stomach", "lever", "bone", "meat"}
    chicken_tokens = {"chicken", "broiler", "cock", "hen"}
    mutton_tokens = {"mutton"}
    fish_tokens = {
        "fish", "rui", "katol", "hilsha", "ilish", "pabda", "tengra",
        "shing", "shrimp", "prawn", "bagda", "golda", "tilapia", "pangas"
    }
    rice_tokens = {"rice", "chal", "chinigura", "miniket", "basmati", "polao"}
    ruti_tokens = {"atta", "maida", "flour", "paratha", "chapati", "ruti"}
    haleem_tokens = {"haleem", "halim", "chatpati", "fuchka"}
    noodles_tokens = {"noodles", "pasta", "macaroni", "ramen", "chowmein"}
    semai_tokens = {"semai", "samai", "vermicelli", "lachcha", "suji", "sagu", "custard"}
    bread_tokens = {"bread", "bun", "toast", "pitha"}
    milk_tokens = {"milk", "dano", "diploma", "powder", "yogurt", "curd", "arla", "pushti", "nido"}
    beverage_tokens = {"coke", "cola", "sprite", "pepsi", "fanta", "7up", "juice", "borhani"}
    snack_tokens = {"chips", "kurkure", "biscuit", "cookies", "chocolate", "chanachur", "nuggets", "sausage"}
    cleaning_tokens = {"vim", "harpic", "detergent", "dishwash", "soap", "tissue", "cleaner", "rin", "wheel", "surf",
                       "toilet"}

    if len(tokens.intersection(haleem_tokens)) > 0:
        return "haleem_cooking"
    if len(tokens.intersection(noodles_tokens)) > 0:
        return "noodles_cooking"
    if len(tokens.intersection(ruti_tokens)) > 0:
        return "ruti_meal"
    if len(tokens.intersection(beef_tokens)) > 0:
        return "beef_cooking"
    if len(tokens.intersection(chicken_tokens)) > 0:
        return "chicken_cooking"
    if len(tokens.intersection(mutton_tokens)) > 0:
        return "mutton_cooking"
    if len(tokens.intersection(fish_tokens)) > 0:
        return "fish_cooking"
    if len(tokens.intersection(rice_tokens)) > 0:
        return "rice_meal"
    if len(tokens.intersection(semai_tokens)) > 0:
        return "dessert"
    if len(tokens.intersection(milk_tokens)) > 0 and len(tokens.intersection(bread_tokens)) > 0:
        return "milk_tea_snacks"
    if len(tokens.intersection(bread_tokens)) > 0:
        return "breakfast"
    if len(tokens.intersection(milk_tokens)) > 0:
        return "milk_breakfast"
    if len(tokens.intersection(beverage_tokens)) > 0 or len(tokens.intersection(snack_tokens)) > 0:
        return "snacks_drink"
    if len(tokens.intersection(cleaning_tokens)) > 0:
        return "cleaning"

    return "general"


def get_rule_profile(intent):
    profiles = {
        "ruti_meal": {
            "must_have": {"dal", "pulse", "egg", "oil", "salt", "sugar", "vegetable", "potato", "onion", "chicken",
                          "beef"},
            "soft_good": {"tea", "coffee", "milk", "butter", "ghee"},
            "bad": {"shampoo", "soap", "detergent", "harpic", "toy", "tissue", "cleaning", "toilet"},
            "same_family_bad": {"atta", "maida", "flour"}
        },
        "haleem_cooking": {
            "must_have": {"beef", "mutton", "chicken", "onion", "ginger", "garlic", "lemon", "coriander", "chilli",
                          "chili", "cucumber", "oil", "salt", "borhani"},
            "soft_good": {"beverage", "drink", "coke", "sprite", "7up"},
            "bad": {"shampoo", "soap", "detergent", "harpic", "sweet", "sugar", "semai", "milk"},
            "same_family_bad": {"haleem", "halim"}
        },
        "noodles_cooking": {
            "must_have": {"egg", "tomato", "sauce", "ketchup", "onion", "chilli", "capsicum", "chicken", "sausage",
                          "oil", "salt"},
            "soft_good": {"cheese", "butter", "mayonnaise"},
            "bad": {"shampoo", "soap", "detergent", "harpic", "sweet", "sugar", "semai"},
            "same_family_bad": {"noodles", "pasta", "macaroni"}
        },
        "milk_tea_snacks": {
            "must_have": {"sugar", "tea", "coffee", "biscuit", "toast", "pitha", "cake", "cookies", "bread", "milk",
                          "dano", "diploma"},
            "soft_good": {"honey", "chocolate", "chips"},
            "bad": {"beef", "fish", "mutton", "onion", "garlic", "ginger", "chilli", "detergent", "soap", "cleaning",
                    "harpic"},
            "same_family_bad": set()
        },
        "beef_cooking": {
            "must_have": {"onion", "garlic", "ginger", "chilli", "chili", "masala", "beef", "garam", "biryani", "oil",
                          "salt", "rice", "chinigura", "potato", "lemon", "borhani", "yogurt", "elachi", "rose",
                          "jafran"},
            "soft_good": {"tomato", "capsicum", "cucumber", "beverage", "drink"},
            "bad": {"sugar", "suji", "semai", "biscuit", "shampoo", "soap", "detergent", "harpic"},
            "same_family_bad": {"salami", "bacon", "sausage"}
        },
        "chicken_cooking": {
            "must_have": {"onion", "garlic", "ginger", "chilli", "chili", "masala", "chicken", "roast", "garam",
                          "biryani", "oil", "salt", "rice", "potato", "lemon", "egg", "yogurt"},
            "soft_good": {"tomato", "capsicum", "cucumber", "borhani"},
            "bad": {"sugar", "suji", "semai", "biscuit", "shampoo", "soap", "detergent", "harpic"},
            "same_family_bad": {"sausage", "nuggets"}
        },
        "fish_cooking": {
            "must_have": {"onion", "garlic", "ginger", "chilli", "chili", "tomato", "coriander", "leaf", "fish",
                          "curry", "masala", "turmeric", "mustard", "oil", "salt", "lemon", "rice"},
            "soft_good": {"capsicum", "pudina", "mint"},
            "bad": {"sugar", "suji", "semai", "biscuit", "shampoo", "soap", "detergent", "beef", "mutton", "sausage"},
            "same_family_bad": set()
        },
        "dessert": {
            "must_have": {"sugar", "milk", "powder", "diploma", "dano", "cardmom", "elachi", "cinnamon", "raisin",
                          "kismis", "ghee", "butter", "yogurt", "dates", "sagu", "semai", "custard", "rose"},
            "soft_good": {"cashew", "kaju", "badam", "pistachio", "nut"},
            "bad": {"onion", "garlic", "ginger", "chilli", "fish", "beef", "chicken", "detergent", "soap", "harpic"},
            "same_family_bad": set()
        },
        "general": {
            "must_have": set(),
            "soft_good": set(),
            "bad": set(),
            "same_family_bad": set()
        }
    }
    return profiles.get(intent, profiles["general"])


def rule_candidate_score(source_name, candidate_name):
    source_intent = infer_product_intent(source_name)
    profile = get_rule_profile(source_intent)

    candidate_tokens = tokenize_name(candidate_name)

    if len(candidate_tokens.intersection(profile["bad"])) > 0:
        return -1.0

    if len(candidate_tokens.intersection(profile["same_family_bad"])) > 0:
        return -0.8

    must_match = len(candidate_tokens.intersection(profile["must_have"]))
    soft_match = len(candidate_tokens.intersection(profile["soft_good"]))

    score = 0.0

    if must_match > 0:
        score += 0.50 + min(must_match, 5) * 0.10

    if soft_match > 0:
        score += 0.20 + min(soft_match, 3) * 0.05

    source_tokens = tokenize_name(source_name)
    common = len(source_tokens.intersection(candidate_tokens))

    if common >= 2:
        score -= 0.30

    return score


def get_rule_seed_candidates(source_name, catalog, top_n=300):
    rows = []
    for idx, row in catalog.iterrows():
        candidate_name = row["item_name"]
        if normalize_name(source_name) == normalize_name(candidate_name):
            continue
        score = rule_candidate_score(source_name, candidate_name)
        if score <= 0:
            continue
        rows.append({
            "candidate_index": int(row["item_index"]),
            "rule_seed_score": float(score),
            "candidate_name": candidate_name,
        })
    rows = sorted(rows, key=lambda x: x["rule_seed_score"], reverse=True)
    return rows[:top_n]


def detect_basket_intent(input_item_names):
    intents = [infer_product_intent(name) for name in input_item_names]
    intent_count = defaultdict(int)
    for intent in intents:
        intent_count[intent] += 1

    if intent_count["haleem_cooking"] > 0: return "haleem_cooking"
    if intent_count["noodles_cooking"] > 0: return "noodles_cooking"
    if intent_count["ruti_meal"] > 0: return "ruti_meal"
    if intent_count["beef_cooking"] > 0: return "beef_cooking"
    if intent_count["chicken_cooking"] > 0: return "chicken_cooking"
    if intent_count["fish_cooking"] > 0: return "fish_cooking"
    if intent_count["dessert"] > 0: return "dessert"
    if intent_count["milk_tea_snacks"] > 0: return "milk_tea_snacks"

    return "general"


def rule_candidate_score_by_intent(source_intent, candidate_name):
    profile = get_rule_profile(source_intent)
    candidate_tokens = tokenize_name(candidate_name)

    if len(candidate_tokens.intersection(profile["bad"])) > 0:
        return -1.0
    if len(candidate_tokens.intersection(profile["same_family_bad"])) > 0:
        return -0.8

    must_match = len(candidate_tokens.intersection(profile["must_have"]))
    soft_match = len(candidate_tokens.intersection(profile["soft_good"]))

    score = 0.0
    if must_match > 0:
        score += 0.50 + min(must_match, 5) * 0.10
    if soft_match > 0:
        score += 0.20 + min(soft_match, 3) * 0.05
    return score


def build_intent_candidate_pool(catalog, top_n=600):
    intents = ["beef_cooking", "chicken_cooking", "mutton_cooking", "fish_cooking", "rice_meal", "ruti_meal",
               "haleem_cooking", "noodles_cooking", "dessert", "milk_tea_snacks", "general"]
    intent_candidate_pool = {}
    for intent in intents:
        rows = []
        for _, row in catalog.iterrows():
            candidate_index = int(row["item_index"])
            candidate_name = row["item_name"]
            score = rule_candidate_score_by_intent(source_intent=intent, candidate_name=candidate_name)
            if score <= 0: continue
            rows.append({
                "candidate_index": candidate_index,
                "candidate_name": candidate_name,
                "rule_seed_score": float(score),
            })
        rows = sorted(rows, key=lambda x: x["rule_seed_score"], reverse=True)
        intent_candidate_pool[intent] = rows[:top_n]
    return intent_candidate_pool