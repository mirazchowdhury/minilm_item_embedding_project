import ast
import json
import math
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from config import (
    BASKET_DATA_PATH,
    BASKET_ONE_HOT_MATRIX_PATH,
    CANDIDATE_POOL_SIZE,
    COOCCURRENCE_MATRIX_PATH,
    COOCCURRENCE_SVD_EMBEDDING_PATH,
    CUSTOMER_COLUMN,
    FEATURE_ITEM_INDEX_PATH,
    ITEM_CATALOG_PATH,
    ITEM_NAME_COLUMN,
    ITEM_POPULARITY_PATH,
    ITEM_USER_MATRIX_PATH,
    ITEM_USER_SVD_EMBEDDING_PATH,
    ITEMS_COLUMN,
    MAIN_DATA_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    SVD_DIM,
    TEXT_EMBEDDING_MATRIX_PATH,
    TOP_K,
    TOPK_RECOMMENDATION_PATH,
)

from business_rule_seed import (
    infer_product_intent,
    rule_candidate_score,
    build_intent_candidate_pool,
)

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def display_name(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def parse_items(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [display_name(x) for x in value if display_name(x)]

    text = str(value).strip()

    if text == "":
        return []

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [display_name(x) for x in parsed if display_name(x)]

    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [display_name(x) for x in parsed if display_name(x)]

    except Exception:
        pass

    if "|" in text:
        return [display_name(x) for x in text.split("|") if display_name(x)]

    if "," in text:
        return [display_name(x) for x in text.split(",") if display_name(x)]

    return [display_name(text)]


def tokenize_product_name(name):
    text = normalize_name(name)

    noise_tokens = {
        "gm", "g", "kg", "ml", "ltr", "liter", "litre",
        "pcs", "pc", "piece", "pieces", "pack", "pkt",
        "box", "bag", "btl", "can", "tin", "jar", "dozen",
        "full", "half", "small", "medium", "large", "mini",
        "special", "regular", "classic", "premium", "super",
        "local", "imported", "fresh", "frozen", "new",
        "the", "and", "with", "without", "combo", "offer",
    }

    tokens = []

    for token in text.split():
        if token in noise_tokens:
            continue

        if token.isdigit():
            continue

        if len(token) <= 1:
            continue

        tokens.append(token)

    return tokens


def token_jaccard(name_a, name_b):
    set_a = set(tokenize_product_name(name_a))
    set_b = set(tokenize_product_name(name_b))

    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0

    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def common_token_count(name_a, name_b):
    set_a = set(tokenize_product_name(name_a))
    set_b = set(tokenize_product_name(name_b))

    return len(set_a.intersection(set_b))


def has_complement_keyword(name):
    tokens = set(tokenize_product_name(name))

    complement_tokens = {
        "masala", "spice", "spices", "onion", "ginger", "garlic",
        "chilli", "chili", "pepper", "salt", "oil", "sauce",
        "rice", "daal", "dal", "lentil", "flour", "atta",
        "bread", "bun", "egg", "milk", "curd", "yogurt",
        "vegetable", "vegetables", "lemon", "tomato", "potato",
        "noodle", "noodles", "pasta", "soup", "mix",
        "battery", "charger", "cable", "refill", "blade",
        "conditioner", "soap", "brush", "paste",
    }

    return len(tokens.intersection(complement_tokens)) > 0

def get_product_family(name):
    tokens = set(tokenize_product_name(name))

    family_groups = {
        "beef": {"beef", "cow", "veal", "koliza", "kima"},
        "chicken": {"chicken", "broiler", "hen", "cock"},
        "fish": {"fish", "rui", "katla", "pangas", "tilapia", "shrimp", "prawn", "bagda", "hilsha", "ilish"},
        "rice": {"rice", "chal", "miniket", "najir", "basmati", "chinigura"},
        "daal": {"daal", "dal", "lentil", "pulse", "moshur", "mug", "boot"},
        "oil": {"oil", "soyabean", "mustard"},
        "spice": {"masala", "spice", "cumin", "turmeric", "chilli", "coriander", "pepper"},
        "vegetable": {"onion", "garlic", "ginger", "potato", "tomato", "brinjal", "chilli", "vegetable"},
        "biscuit": {"biscuit", "cookies", "cracker"},
        "snacks": {"chips", "chanachur", "kurkure", "snacks"},
        "dairy": {"milk", "curd", "yogurt", "doi", "cheese", "butter"},
        "personal_care": {"soap", "shampoo", "conditioner", "lotion", "cream"},
        "oral_care": {"toothpaste", "toothbrush", "brush"},
        "cleaning": {"detergent", "harpic", "cleaner", "powder", "liquid"},
    }

    matched_families = []

    for family_name, family_tokens in family_groups.items():
        if len(tokens.intersection(family_tokens)) > 0:
            matched_families.append(family_name)

    return set(matched_families)


def is_same_family_product(source_name, candidate_name):
    source_family = get_product_family(source_name)
    candidate_family = get_product_family(candidate_name)

    if len(source_family) == 0 or len(candidate_family) == 0:
        return False

    return len(source_family.intersection(candidate_family)) > 0


def get_min_pair_required(source_count):
    if source_count >= 1000:
        return 5

    if source_count >= 500:
        return 4

    if source_count >= 100:
        return 3

    if source_count >= 30:
        return 2

    return 1


def clip_score(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(float(value), max_value))



def is_same_type_product(source_name, candidate_name):
    jaccard = token_jaccard(source_name, candidate_name)
    common_count = common_token_count(source_name, candidate_name)

    if has_complement_keyword(candidate_name):
        return False

    if jaccard >= 0.35:
        return True

    if common_count >= 2:
        return True

    return False

def infer_item_group(name):
    # Product name দেখে broad item group detect করা হচ্ছে
    text = normalize_name(name)
    tokens = set(tokenize_product_name(name))

    group_rules = [
        ("raw_meat", {
            "beef", "mutton", "chicken", "cock", "hen", "duck",
            "meat", "bone", "koliza", "lever", "stomach", "brain",
            "breast", "wings", "drumstick", "drumsticks", "neck"
        }),
        ("processed_meat", {
            "sausage", "salami", "bacon", "kabab", "nuggets",
            "nugget", "samosa", "samusa", "singara", "roll",
            "patty", "meatball", "meatballs", "popcorn", "lollipop"
        }),
        ("raw_fish", {
            "fish", "shrimp", "prawn", "rui", "katol", "hilsha",
            "ilish", "pabda", "pangas", "tilapia", "koi", "magur",
            "shing", "tengra", "boal", "mola", "kachki", "bele",
            "koral", "rupchanda", "golda", "bagda", "crab"
        }),
        ("vegetable", {
            "onion", "garlic", "ginger", "chilli", "chili", "potato",
            "tomato", "brinjal", "cucumber", "carrot", "pumpkin",
            "potol", "parboil", "vendi", "ladies", "finger", "lau",
            "papaya", "cabbage", "cauliflower", "capsicum", "lemon",
            "coriander", "leaf", "pudina", "mint", "shak", "beans",
            "radish", "vegetable"
        }),
        ("fruit", {
            "apple", "banana", "mango", "grapes", "grape", "malta",
            "orange", "pomegranate", "guava", "dragon", "watermelon",
            "pineapple", "papaya", "naspati", "latkon", "jamrul",
            "sofeda", "jumbura", "amra", "kamranga", "fruit"
        }),
        ("spice", {
            "masala", "spice", "spices", "cumin", "turmeric", "chilli",
            "chili", "coriander", "pepper", "garam", "biryani", "biriyani",
            "tehari", "roast", "korma", "kabab", "curry", "mustard",
            "panch", "foron", "cardmom", "elachi", "cinnamon", "clove",
            "lobongo", "methi", "bay", "leaf", "tasting", "salt", "ginger",
            "garlic", "powder"
        }),
        ("staple", {
            "rice", "chal", "chinigura", "miniket", "najir", "basmati",
            "atta", "maida", "flour", "suji", "semolina", "sugar",
            "salt", "chira", "muri", "puffed", "vermicelli", "semai",
            "samai", "lachcha", "noodles", "pasta", "macaroni"
        }),
        ("pulse", {
            "daal", "dal", "pulse", "lentil", "mosur", "mushur",
            "mug", "moog", "boot", "chola", "khesari", "dubli",
            "maskalai", "mixed", "motor"
        }),
        ("oil", {
            "oil", "soyabean", "mustard", "sunflower", "olive",
            "rice", "bran", "ghee", "butter", "margarine"
        }),
        ("dairy", {
            "milk", "curd", "yogurt", "doi", "cheese", "paneer",
            "butter", "ghee", "laban", "matha", "cream"
        }),
        ("bakery", {
            "bread", "bun", "toast", "cake", "biscuit", "cookies",
            "cracker", "croissant", "muffin", "wafer", "digestive",
            "rusk", "dry", "sandwich"
        }),
        ("snack", {
            "chips", "chanachur", "kurkure", "lays", "snacks",
            "chocolate", "candy", "gum", "lollipop", "popcorn",
            "badam", "nut", "peanut", "cashew", "pistachio",
            "papri", "nimki", "murali", "vaja"
        }),
        ("beverage", {
            "coke", "cola", "sprite", "pepsi", "fanta", "7up",
            "drink", "juice", "water", "tea", "coffee", "borhani",
            "labang", "milk", "latte", "nescafe"
        }),
        ("sauce", {
            "sauce", "ketchup", "vinegar", "mayonnaise", "mayo",
            "kasundi", "pickle", "chutney", "soy", "oyster",
            "bbq", "dressing"
        }),
        ("cleaning", {
            "detergent", "soap", "dishwash", "harpic", "lizol",
            "cleaner", "vim", "rin", "wheel", "surf", "toilet",
            "laundry", "powder", "phenyl", "bleach"
        }),
        ("personal_care", {
            "shampoo", "conditioner", "toothpaste", "toothbrush",
            "face", "wash", "cream", "lotion", "oil", "hair",
            "dove", "sunsilk", "pantene", "ponds", "colgate",
            "sensodyne", "pepsodent", "vaseline", "nivea"
        }),
        ("household", {
            "tissue", "napkin", "towel", "foil", "container",
            "box", "basket", "jar", "mug", "plate", "bowl",
            "spoon", "glass", "tray", "mat", "brush", "hanger",
            "bucket", "jug", "pot", "pan", "karai", "tawa"
        }),
        ("kitchen_appliance", {
            "rice", "cooker", "blender", "mixer", "kettle", "beater",
            "dryer", "hair", "sandwich", "maker", "electric",
            "infrared", "induction", "pressure", "cooker"
        }),
        ("baby", {
            "baby", "feeding", "bottle", "nipple", "diaper",
            "cerelac", "lactogen", "nan", "nido", "johnsons",
            "kodomo", "huggies", "pampers"
        }),
        ("toy_stationery", {
            "toy", "car", "ball", "pencil", "pen", "khata",
            "notebook", "eraser", "sharpner", "stapler", "card",
            "game", "ludu", "monopoly", "puzzle", "doll"
        }),
        ("clothing_beauty", {
            "lipstick", "eyeliner", "mascara", "powder", "foundation",
            "nail", "comb", "clip", "band", "bra", "vest", "panty",
            "trouser", "tshirt", "shirt", "orna", "hijab", "tupi"
        }),
    ]

    matched_groups = []

    for group, keywords in group_rules:
        if len(tokens.intersection(keywords)) > 0:
            matched_groups.append(group)

    if len(matched_groups) == 0:
        return "unknown"

    # Special correction
    if "rice" in tokens and ("cooker" not in tokens and "spoon" not in tokens and "bowl" not in tokens and "plate" not in tokens):
        return "staple"

    if "hair" in tokens and ("dryer" in tokens or "straightener" in tokens):
        return "personal_care"

    return matched_groups[0]

def infer_item_intent(name):
    text = normalize_name(name)
    tokens = set(tokenize_product_name(name))

    dessert_base_tokens = {
        "suji", "semolina", "sagu", "semai", "samai", "vermicelli",
        "lachcha", "falooda", "custard", "firni", "kheer", "halua",
        "pudding", "jelly"
    }

    sweetener_tokens = {
        "sugar", "chini", "molasses", "misri", "candy"
    }

    milk_tokens = {
        "milk", "powder", "dano", "diploma", "marks", "aarong",
        "farm", "fresh", "milk", "condensed", "cream"
    }

    dessert_addon_tokens = {
        "raisin", "kismis", "cardmom", "elachi", "cinnamon",
        "daruchine", "kewra", "rose", "water", "ghee", "butter",
        "cashew", "kaju", "badam", "nut", "pistachio", "saffron",
        "jafran"
    }

    cooking_base_tokens = {
        "rice", "atta", "maida", "flour", "dal", "daal", "pulse",
        "oil", "salt"
    }

    meat_cooking_tokens = {
        "beef", "mutton", "chicken", "fish", "shrimp", "prawn"
    }

    if len(tokens.intersection(dessert_base_tokens)) > 0:
        return "dessert_base"

    if len(tokens.intersection(sweetener_tokens)) > 0:
        return "sweetener"

    if len(tokens.intersection(milk_tokens)) > 0 and "milk" in tokens:
        return "milk"

    if len(tokens.intersection(dessert_addon_tokens)) > 0:
        return "dessert_addon"

    if len(tokens.intersection(meat_cooking_tokens)) > 0:
        return "meat_fish_base"

    if len(tokens.intersection(cooking_base_tokens)) > 0:
        return "daily_cooking_base"

    return "general"
def infer_cooking_role(name):
    tokens = set(tokenize_product_name(name))

    sweet_tokens = {
        "sugar", "chini", "molasses", "misri", "candy",
        "suji", "sagu", "semai", "samai", "vermicelli",
        "custard", "falooda", "kheer", "firni", "jelly",
        "cake", "chocolate", "biscuit", "cookies"
    }

    meat_aromatic_tokens = {
        "onion", "garlic", "ginger", "chilli", "chili",
        "green", "capsicum", "tomato", "lemon", "coriander",
        "leaf", "pudina", "mint"
    }

    meat_bad_veg_tokens = {
        "cucumber", "lau", "gourd", "papaya", "pumpkin",
        "potol", "vendi", "ladies", "finger", "jhinga",
        "cabbage", "cauliflower", "carrot", "radish"
    }

    rice_tokens = {
        "rice", "chal", "chinigura", "miniket", "najir",
        "basmati", "polao", "pulao"
    }

    oil_tokens = {
        "oil", "soyabean", "mustard", "sunflower", "ghee"
    }

    meat_masala_tokens = {
        "masala", "spice", "spices", "biryani", "biriyani",
        "beef", "meat", "mutton", "chicken", "korma", "roast",
        "tehari", "kabab", "tikka", "garam", "cumin",
        "coriander", "turmeric", "chilli", "chili", "pepper",
        "cardmom", "elachi", "cinnamon", "clove", "lobongo",
        "bay", "leaf", "methi"
    }

    beverage_tokens = {
        "coke", "cola", "sprite", "pepsi", "fanta", "7up",
        "borhani", "labang", "drink", "juice", "water"
    }

    if len(tokens.intersection(sweet_tokens)) > 0:
        return "sweet"

    if len(tokens.intersection(meat_aromatic_tokens)) > 0:
        return "meat_aromatic"

    if len(tokens.intersection(meat_bad_veg_tokens)) > 0:
        return "meat_bad_vegetable"

    if len(tokens.intersection(rice_tokens)) > 0:
        return "rice"

    if len(tokens.intersection(oil_tokens)) > 0:
        return "oil"

    if len(tokens.intersection(meat_masala_tokens)) > 0:
        return "meat_masala"

    if len(tokens.intersection(beverage_tokens)) > 0:
        return "beverage"

    return "general"


def intent_compatibility_score(source_name, candidate_name):
    source_intent = infer_item_intent(source_name)
    candidate_intent = infer_item_intent(candidate_name)

    source_group = infer_item_group(source_name)
    candidate_group = infer_item_group(candidate_name)

    source_role = infer_cooking_role(source_name)
    candidate_role = infer_cooking_role(candidate_name)

    if source_intent == "meat_fish_base" or source_group in {"raw_meat", "raw_fish"}:
        if candidate_role == "meat_masala":
            return 0.45

        if candidate_role == "meat_aromatic":
            return 0.42

        if candidate_role == "rice":
            return 0.35

        if candidate_role == "oil":
            return 0.30

        if candidate_role == "beverage":
            return 0.18

        if candidate_role == "sweet":
            return -1.00

        if candidate_role == "meat_bad_vegetable":
            return -0.70

        if candidate_group in {
            "cleaning",
            "personal_care",
            "toy_stationery",
            "clothing_beauty",
            "baby",
            "household",
            "fruit",
            "bakery",
            "snack",
            "dairy",
        }:
            return -1.00

        if candidate_group in {"spice", "vegetable", "staple", "oil", "sauce", "beverage"}:
            return 0.10

        return -0.40

    if source_intent == "dessert_base":
        good_intents = {
            "sweetener",
            "milk",
            "dessert_addon",
            "dessert_base"
        }

        good_groups = {
            "dairy",
            "staple",
            "snack"
        }

        bad_groups = {
            "cleaning",
            "personal_care",
            "toy_stationery",
            "clothing_beauty",
            "household",
            "raw_meat",
            "raw_fish",
            "processed_meat",
            "vegetable"
        }

        if candidate_intent in good_intents:
            return 0.35

        if candidate_group in good_groups:
            return 0.12

        if candidate_group in bad_groups:
            return -0.80

        return -0.25

    return 0.0


def get_allowed_candidate_groups(source_group):
    # Source group অনুযায়ী কোন group recommendation হিসেবে safe সেটা define করা হচ্ছে
    rules = {
        "raw_meat": {
            "spice", "vegetable", "staple", "pulse", "oil", "sauce", "dairy"
        },
        "raw_fish": {
            "spice", "vegetable", "staple", "pulse", "oil", "sauce"
        },
        "processed_meat": {
            "sauce", "bakery", "beverage", "snack", "staple", "dairy"
        },
        "vegetable": {
            "vegetable", "spice", "raw_meat", "raw_fish", "staple", "pulse", "oil"
        },
        "fruit": {
            "fruit", "dairy", "snack", "beverage"
        },
        "spice": {
            "raw_meat", "raw_fish", "vegetable", "staple", "pulse", "oil", "sauce"
        },
        "staple": {
            "pulse", "vegetable", "spice", "raw_meat", "raw_fish", "oil", "dairy", "sauce"
        },
        "pulse": {
            "staple", "vegetable", "spice", "oil", "raw_meat", "raw_fish"
        },
        "oil": {
            "staple", "pulse", "vegetable", "spice", "raw_meat", "raw_fish"
        },
        "dairy": {
            "bakery", "staple", "snack", "beverage", "fruit"
        },
        "bakery": {
            "dairy", "snack", "beverage", "sauce", "processed_meat"
        },
        "snack": {
            "beverage", "snack", "bakery", "dairy"
        },
        "beverage": {
            "snack", "bakery", "dairy", "beverage"
        },
        "sauce": {
            "processed_meat", "raw_meat", "raw_fish", "bakery", "staple", "snack"
        },
        "cleaning": {
            "cleaning", "household"
        },
        "personal_care": {
            "personal_care", "cleaning"
        },
        "household": {
            "household", "cleaning", "kitchen_appliance"
        },
        "kitchen_appliance": {
            "household", "staple", "oil", "spice", "kitchen_appliance"
        },
        "baby": {
            "baby", "dairy"
        },
        "toy_stationery": {
            "toy_stationery"
        },
        "clothing_beauty": {
            "clothing_beauty", "personal_care"
        },
        "unknown": {
            "staple", "vegetable", "spice", "pulse", "oil", "dairy", "bakery", "snack"
        },
    }

    return rules.get(source_group, rules["unknown"])


def is_allowed_by_group(source_name, candidate_name):
    source_intent = infer_item_intent(source_name)
    source_group = infer_item_group(source_name)
    candidate_group = infer_item_group(candidate_name)
    candidate_role = infer_cooking_role(candidate_name)

    if source_intent == "meat_fish_base" or source_group in {"raw_meat", "raw_fish"}:
        allowed_roles = {
            "meat_masala",
            "meat_aromatic",
            "rice",
            "oil",
            "beverage"
        }

        if candidate_role in allowed_roles:
            return True

        allowed_groups = {
            "spice",
            "vegetable",
            "staple",
            "oil",
            "sauce",
            "beverage"
        }

        if candidate_group in allowed_groups and candidate_role not in {"sweet", "meat_bad_vegetable"}:
            return True

        return False

    if source_intent == "dessert_base":
        allowed_intents = {
            "sweetener",
            "milk",
            "dessert_addon",
            "dessert_base"
        }

        allowed_groups = {
            "dairy",
            "staple",
            "snack"
        }

        candidate_intent = infer_item_intent(candidate_name)

        if candidate_intent in allowed_intents:
            return True

        if candidate_group in allowed_groups:
            return True

        return False

    allowed_groups = get_allowed_candidate_groups(source_group)

    return candidate_group in allowed_groups


def is_bad_same_type(source_name, candidate_name):
    source_group = infer_item_group(source_name)
    candidate_group = infer_item_group(candidate_name)

    if source_group != candidate_group:
        return False

    # এই group গুলোর মধ্যে same group recommendation naturally useful হতে পারে
    allowed_same_group = {
        "vegetable",
        "fruit",
        "snack",
        "beverage",
        "cleaning",
        "personal_care",
        "household",
        "toy_stationery",
        "clothing_beauty",
    }

    if source_group in allowed_same_group:
        return False

    if token_jaccard(source_name, candidate_name) >= 0.25:
        return True

    if common_token_count(source_name, candidate_name) >= 1:
        return True

    return False


def get_reliable_min_pair(source_count):
    # Popular item হলে pair count বেশি না হলে recommendation reliable না
    if source_count >= 1500:
        return 25

    if source_count >= 1000:
        return 18

    if source_count >= 500:
        return 10

    if source_count >= 200:
        return 6

    if source_count >= 80:
        return 4

    if source_count >= 30:
        return 3

    return 2


def normalize_score_value(value):
    if pd.isna(value):
        return 0.0

    value = float(value)

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return value

def safe_divide(a, b):
    if b == 0:
        return 0.0

    return float(a) / float(b)


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return matrix / norms


def load_catalog():
    if not ITEM_CATALOG_PATH.exists():
        raise FileNotFoundError(f"Item catalog not found: {ITEM_CATALOG_PATH}")

    catalog = pd.read_csv(ITEM_CATALOG_PATH)

    if "item_name" not in catalog.columns:
        raise ValueError("item_catalog_from_basket.csv must contain item_name column")

    if "purchase_count" not in catalog.columns:
        catalog["purchase_count"] = 1

    catalog["item_name"] = catalog["item_name"].apply(display_name)
    catalog["normalized_item_name"] = catalog["item_name"].apply(normalize_name)

    # Important:
    # Duplicate drop করা যাবে না, কারণ MiniLM embedding already original catalog row order অনুযায়ী তৈরি হয়েছে
    # Catalog rows এবং embedding rows same থাকতে হবে

    catalog = catalog.reset_index(drop=True)

    catalog["item_index"] = np.arange(len(catalog))
    catalog["item_id"] = catalog["item_index"] + 1

    catalog = catalog[
        [
            "item_index",
            "item_id",
            "item_name",
            "normalized_item_name",
            "purchase_count",
        ]
    ].copy()

    catalog.to_csv(FEATURE_ITEM_INDEX_PATH, index=False)

    return catalog

def load_basket_dataframe():
    if not BASKET_DATA_PATH.exists():
        raise FileNotFoundError(f"Basket file not found: {BASKET_DATA_PATH}")

    basket_df = pd.read_csv(BASKET_DATA_PATH)

    if ITEMS_COLUMN not in basket_df.columns:
        raise ValueError(f"Basket column not found: {ITEMS_COLUMN}")

    basket_df["basket_items_raw"] = basket_df[ITEMS_COLUMN].apply(parse_items)
    basket_df["basket_items"] = basket_df["basket_items_raw"].apply(
        lambda items: sorted(set([normalize_name(x) for x in items if normalize_name(x)]))
    )

    basket_df["basket_size_clean"] = basket_df["basket_items"].apply(len)
    basket_df = basket_df[basket_df["basket_size_clean"] >= 2].copy()

    return basket_df


def build_one_hot_and_cooccurrence(basket_df, catalog):
    item_to_index = dict(
        zip(catalog["normalized_item_name"], catalog["item_index"])
    )

    rows = []
    cols = []
    data = []

    item_counter = Counter()
    pair_counter = Counter()

    valid_basket_count = 0

    for basket_row_index, basket_items in enumerate(basket_df["basket_items"].tolist()):
        item_indices = []

        for item_name in basket_items:
            if item_name in item_to_index:
                item_indices.append(item_to_index[item_name])

        item_indices = sorted(set(item_indices))

        if len(item_indices) < 2:
            continue

        current_row = valid_basket_count
        valid_basket_count += 1

        for item_index in item_indices:
            rows.append(current_row)
            cols.append(item_index)
            data.append(1)
            item_counter[item_index] += 1

        for item_a, item_b in combinations(item_indices, 2):
            pair_counter[(item_a, item_b)] += 1

    one_hot_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(valid_basket_count, len(catalog)),
        dtype=np.float32,
    )

    cooccurrence_matrix = one_hot_matrix.T @ one_hot_matrix
    cooccurrence_matrix.setdiag(0)
    cooccurrence_matrix.eliminate_zeros()

    save_npz(BASKET_ONE_HOT_MATRIX_PATH, one_hot_matrix)
    save_npz(COOCCURRENCE_MATRIX_PATH, cooccurrence_matrix)

    print(f"Saved one hot matrix: {BASKET_ONE_HOT_MATRIX_PATH}")
    print(f"One hot matrix shape: {one_hot_matrix.shape}")
    print(f"Saved co occurrence matrix: {COOCCURRENCE_MATRIX_PATH}")
    print(f"Co occurrence matrix shape: {cooccurrence_matrix.shape}")

    return one_hot_matrix, cooccurrence_matrix, item_counter, pair_counter, valid_basket_count


def build_item_user_matrix(catalog):
    if not MAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Main dataset not found: {MAIN_DATA_PATH}")

    main_df = pd.read_csv(MAIN_DATA_PATH)

    if CUSTOMER_COLUMN not in main_df.columns:
        raise ValueError(f"Customer column not found: {CUSTOMER_COLUMN}")

    if ITEM_NAME_COLUMN not in main_df.columns:
        raise ValueError(f"Item column not found: {ITEM_NAME_COLUMN}")

    catalog_item_to_index = dict(
        zip(catalog["normalized_item_name"], catalog["item_index"])
    )

    main_df["normalized_item_name"] = main_df[ITEM_NAME_COLUMN].apply(normalize_name)
    main_df = main_df[main_df["normalized_item_name"].isin(catalog_item_to_index)].copy()

    customer_values = main_df[CUSTOMER_COLUMN].astype(str).fillna("unknown").tolist()
    unique_customers = sorted(set(customer_values))
    customer_to_index = {customer: index for index, customer in enumerate(unique_customers)}

    rows = []
    cols = []
    data = []

    for _, row in main_df.iterrows():
        item_index = catalog_item_to_index.get(row["normalized_item_name"])
        customer_index = customer_to_index.get(str(row[CUSTOMER_COLUMN]))

        if item_index is None or customer_index is None:
            continue

        quantity = 1.0

        if "quantity" in main_df.columns:
            try:
                quantity = float(row["quantity"])
            except Exception:
                quantity = 1.0

        if quantity <= 0:
            quantity = 1.0

        rows.append(item_index)
        cols.append(customer_index)
        data.append(quantity)

    item_user_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(catalog), len(unique_customers)),
        dtype=np.float32,
    )

    save_npz(ITEM_USER_MATRIX_PATH, item_user_matrix)

    print(f"Saved item user matrix: {ITEM_USER_MATRIX_PATH}")
    print(f"Item user matrix shape: {item_user_matrix.shape}")

    return item_user_matrix


def create_svd_embedding(sparse_matrix, output_path, matrix_name):
    max_dim = min(sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1, SVD_DIM)

    if max_dim < 2:
        raise ValueError(f"{matrix_name} is too small for SVD")

    svd = TruncatedSVD(
        n_components=max_dim,
        random_state=42,
        n_iter=10,
    )

    embedding = svd.fit_transform(sparse_matrix)
    embedding = normalize(embedding).astype(np.float32)

    np.save(output_path, embedding)

    explained = float(svd.explained_variance_ratio_.sum())

    print(f"Saved {matrix_name} SVD embedding: {output_path}")
    print(f"{matrix_name} embedding shape: {embedding.shape}")
    print(f"{matrix_name} explained variance ratio: {round(explained, 4)}")

    return embedding


def load_text_embedding(catalog):
    if not TEXT_EMBEDDING_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"MiniLM embedding file not found: {TEXT_EMBEDDING_MATRIX_PATH}"
        )

    text_embedding = np.load(TEXT_EMBEDDING_MATRIX_PATH).astype(np.float32)

    if text_embedding.shape[0] != len(catalog):
        raise ValueError(
            f"MiniLM embedding row count does not match catalog. "
            f"MiniLM rows: {text_embedding.shape[0]}, catalog rows: {len(catalog)}"
        )

    text_embedding = l2_normalize(text_embedding).astype(np.float32)

    return text_embedding


def get_sparse_pair_count(cooccurrence_matrix, item_a, item_b):
    return float(cooccurrence_matrix[item_a, item_b])


def get_top_sparse_neighbors(cooccurrence_matrix, source_index, top_n):
    row = cooccurrence_matrix.getrow(source_index)

    if row.nnz == 0:
        return []

    candidate_indices = row.indices
    candidate_scores = row.data

    order = np.argsort(candidate_scores)[::-1][:top_n]

    return candidate_indices[order].tolist()


def get_top_similarity_indices(embedding, source_index, top_n):
    source_vector = embedding[source_index]
    scores = embedding @ source_vector

    order = np.argsort(scores)[::-1]

    result = []

    for index in order:
        if index == source_index:
            continue

        result.append(int(index))

        if len(result) >= top_n:
            break

    return result


def get_similarity_score(embedding, source_index, candidate_index):
    return float(np.dot(embedding[source_index], embedding[candidate_index]))


def create_popularity_table(catalog, item_counter):
    rows = []

    max_count = max(item_counter.values()) if len(item_counter) > 0 else 1

    for _, row in catalog.iterrows():
        item_index = int(row["item_index"])
        count = int(item_counter.get(item_index, row["purchase_count"]))

        rows.append(
            {
                "item_index": item_index,
                "item_name": row["item_name"],
                "purchase_count": count,
                "popularity_score": safe_divide(count, max_count),
            }
        )

    popularity_df = pd.DataFrame(rows)
    popularity_df = popularity_df.sort_values("purchase_count", ascending=False)
    popularity_df.to_csv(ITEM_POPULARITY_PATH, index=False)

    print(f"Saved popularity table: {ITEM_POPULARITY_PATH}")

    return popularity_df



def generate_final_recommendations(
    catalog,
    item_counter,
    cooccurrence_matrix,
    item_user_embedding,
    cooccurrence_embedding,
    text_embedding,
    total_baskets,
):
    popularity_map = {}

    max_count = max(item_counter.values()) if len(item_counter) > 0 else 1

    for item_index in range(len(catalog)):
        popularity_map[item_index] = safe_divide(
            item_counter.get(item_index, 0),
            max_count,
        )

    all_rows = []

    print("Building intent candidate pool")
    intent_candidate_pool = build_intent_candidate_pool(
        catalog=catalog,
        top_n=CANDIDATE_POOL_SIZE,
    )
    print("Intent candidate pool ready")
    print("Generating business rule seeded top recommendations")

    for source_index in range(len(catalog)):
        source_name = catalog.loc[source_index, "item_name"]
        source_count = item_counter.get(source_index, 0)
        source_intent = infer_product_intent(source_name)

        candidate_set = set()
        candidate_rule_score_map = {}

        # First priority, business rule seed candidate
        rule_seed_rows = intent_candidate_pool.get(source_intent, [])

        for seed_row in rule_seed_rows:
            candidate_index = seed_row["candidate_index"]

            if candidate_index == source_index:
                continue

            candidate_set.add(candidate_index)
            candidate_rule_score_map[candidate_index] = seed_row["rule_seed_score"]

        # Second priority, real basket co purchase candidate
        co_candidates = get_top_sparse_neighbors(
            cooccurrence_matrix,
            source_index,
            CANDIDATE_POOL_SIZE,
        )

        for candidate_index in co_candidates:
            candidate_set.add(candidate_index)

        # Third priority, embedding candidate, only support
        for candidate_index in get_top_similarity_indices(
            item_user_embedding,
            source_index,
            30,
        ):
            candidate_set.add(candidate_index)

        for candidate_index in get_top_similarity_indices(
            cooccurrence_embedding,
            source_index,
            30,
        ):
            candidate_set.add(candidate_index)

        candidate_scores = []

        for candidate_index in candidate_set:
            if candidate_index == source_index:
                continue

            candidate_name = catalog.loc[candidate_index, "item_name"]
            candidate_count = item_counter.get(candidate_index, 0)

            rule_seed_score = candidate_rule_score_map.get(
                candidate_index,
                rule_candidate_score(source_name, candidate_name),
            )

            # Rule score negative হলে candidate বাদ
            if rule_seed_score < 0:
                continue

            pair_count = get_sparse_pair_count(
                cooccurrence_matrix,
                source_index,
                candidate_index,
            )

            item_user_similarity = normalize_score_value(
                get_similarity_score(
                    item_user_embedding,
                    source_index,
                    candidate_index,
                )
            )

            cooccurrence_similarity = normalize_score_value(
                get_similarity_score(
                    cooccurrence_embedding,
                    source_index,
                    candidate_index,
                )
            )

            text_similarity = normalize_score_value(
                get_similarity_score(
                    text_embedding,
                    source_index,
                    candidate_index,
                )
            )

            co_purchase_score = safe_divide(pair_count, source_count)
            reverse_confidence = safe_divide(pair_count, candidate_count)

            cooccurrence_cosine = safe_divide(
                pair_count,
                math.sqrt(max(source_count, 1) * max(candidate_count, 1)),
            )

            # ... (Inside generate_final_recommendations function, around line 560) ...

            pair_strength = safe_divide(
                math.log1p(pair_count),
                math.log1p(max(source_count, 1)),
            )

            # ... Ager code thik thakbe ...
            popularity_score = popularity_map.get(candidate_index, 0.0)

            # Updated strict score for matrix generation
            if rule_seed_score < 0:
                continue

            final_score = (
                    0.70 * rule_seed_score
                    + 0.15 * text_similarity
                    + 0.10 * popularity_score
                    + 0.05 * co_purchase_score
            )

            recommendation_type = "rule_seed"

            if pair_count > 0:
                recommendation_type = "rule_seed_with_copurchase"

            candidate_scores.append(
                {
                    "source_item_index": source_index,
                    "source_item_name": source_name,
                    "source_intent": source_intent,
                    "recommended_item_index": candidate_index,
                    "recommended_item_name": candidate_name,
                    "recommended_intent": infer_product_intent(candidate_name),
                    "recommendation_type": recommendation_type,
                    "pair_count": int(pair_count),
                    "source_count": int(source_count),
                    "candidate_count": int(candidate_count),
                    "rule_seed_score": rule_seed_score,
                    "co_purchase_score": co_purchase_score,
                    "cooccurrence_cosine": cooccurrence_cosine,
                    "pair_strength": pair_strength,
                    "reverse_confidence": reverse_confidence,
                    "item_user_similarity": item_user_similarity,
                    "cooccurrence_similarity": cooccurrence_similarity,
                    "text_similarity": text_similarity,
                    "popularity_score": popularity_score,
                    "final_score": final_score,
                }
            )

        if len(candidate_scores) == 0:
            continue

        candidate_df = pd.DataFrame(candidate_scores)

        candidate_df = candidate_df.sort_values(
            by=[
                "final_score",
                "rule_seed_score",
                "pair_count",
                "co_purchase_score",
                "popularity_score",
            ],
            ascending=[False, False, False, False, False],
        )

        # Same recommendation name repeat avoid
        selected_rows = []
        used_names = set()

        for _, row in candidate_df.iterrows():
            candidate_name_norm = normalize_name(row["recommended_item_name"])

            if candidate_name_norm in used_names:
                continue

            selected_rows.append(row.to_dict())
            used_names.add(candidate_name_norm)

            if len(selected_rows) >= TOP_K:
                break

        for rank, row_dict in enumerate(selected_rows, start=1):
            row_dict["rank"] = rank
            all_rows.append(row_dict)

    recommendation_df = pd.DataFrame(all_rows)

    recommendation_df.to_csv(TOPK_RECOMMENDATION_PATH, index=False)

    print(f"Saved final recommendation file: {TOPK_RECOMMENDATION_PATH}")
    print(f"Total recommendation rows: {len(recommendation_df)}")

    return recommendation_df

def build_recommendations():
    ensure_dirs()

    catalog = load_catalog()

    basket_df = load_basket_dataframe()

    one_hot_matrix, cooccurrence_matrix, item_counter, pair_counter, total_baskets = (
        build_one_hot_and_cooccurrence(
            basket_df,
            catalog,
        )
    )

    item_user_matrix = build_item_user_matrix(catalog)

    item_user_embedding = create_svd_embedding(
        item_user_matrix,
        ITEM_USER_SVD_EMBEDDING_PATH,
        "item user",
    )

    cooccurrence_embedding = create_svd_embedding(
        cooccurrence_matrix,
        COOCCURRENCE_SVD_EMBEDDING_PATH,
        "co occurrence",
    )

    text_embedding = load_text_embedding(catalog)

    create_popularity_table(catalog, item_counter)

    recommendation_df = generate_final_recommendations(
        catalog=catalog,
        item_counter=item_counter,
        cooccurrence_matrix=cooccurrence_matrix,
        item_user_embedding=item_user_embedding,
        cooccurrence_embedding=cooccurrence_embedding,
        text_embedding=text_embedding,
        total_baskets=total_baskets,
    )

    return recommendation_df


if __name__ == "__main__":
    build_recommendations()
