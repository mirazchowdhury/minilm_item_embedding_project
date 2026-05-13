import re


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_name(name):
    return set(normalize_text(name).split())


def has_any(tokens, keywords):
    return len(tokens.intersection(keywords)) > 0


def get_product_family(name):
    tokens = tokenize_name(name)
    text = normalize_text(name)

    meat_tokens = {
        "beef", "cow", "mutton", "meat", "koliza", "lever",
        "kima", "keema", "brain", "stomach", "bone", "gosht"
    }

    chicken_tokens = {
        "chicken", "cock", "broiler", "breast", "wings",
        "drumstick", "lollipop"
    }

    fish_tokens = {
        "fish", "rui", "hilsha", "shrimp", "pabda", "tengra",
        "katol", "boal", "pangas", "koi", "magur", "shing",
        "mrigel", "tilapia", "telapia", "golda", "bagda",
        "poa", "ayir", "bele", "koral", "laitta", "mola",
        "rupchanda", "rupchada", "piali", "bata", "koral",
        "ilish", "chitol", "surma", "crab", "kaikka"
    }

    rice_tokens = {
        "rice", "chinigura", "basmati", "miniket",
        "najirshail", "polao", "aromatic", "chal"
    }

    cooking_tokens = {
        "onion", "garlic", "ginger", "chilli", "chili",
        "tomato", "potato", "coriander", "lemon", "capsicum",
        "morich", "peyaj", "roshun", "ada", "cucumber",
        "pudina", "mint"
    }

    spice_tokens = {
        "masala", "cumin", "turmeric", "coriander", "pepper",
        "cinnamon", "cardmom", "cardamom", "clove", "panchforon",
        "panch", "foron", "methi", "bay", "leaf", "powder",
        "biryani", "biriyani", "garam", "radhuni", "shan",
        "savory", "tehari", "kacchi", "korma", "roast",
        "curry", "kabab", "kasundi", "paprika", "oregano",
        "chilli", "chili"
    }

    oil_tokens = {
        "oil", "soyabean", "soyabin", "soya", "mustard", "ghee"
    }

    dairy_tokens = {
        "milk", "butter", "cheese", "yogurt", "curd",
        "doi", "cream", "paneer", "dano", "diploma", "aarong",
        "matha", "labang", "lassi"
    }

    dessert_tokens = {
        "sugar", "semai", "samai", "suji", "sagu", "vermicelli",
        "custard", "falooda", "firni", "kheer", "jelly",
        "raisin", "kismis", "dates", "lachcha", "lacha",
        "shemai", "semolina", "halua", "pudding"
    }

    bakery_tokens = {
        "bread", "toast", "bun", "cake", "biscuit",
        "cookies", "cracker", "paratha", "roti", "bakorkhani"
    }

    beverage_tokens = {
        "coke", "cola", "sprite", "pepsi", "juice",
        "borhani", "fanta", "dew", "7up", "mojo",
        "water", "drink", "coffee", "tea"
    }

    snacks_tokens = {
        "chips", "chanachur", "kurkure", "lays", "pringles",
        "snacks", "nuggets", "sausage", "fries", "samosa",
        "singara", "spring", "roll", "popcorn", "noodles",
        "ramen", "pasta", "macaroni", "burger", "pizza",
        "sauce", "mayonnaise"
    }

    cleaning_tokens = {
        "vim", "harpic", "detergent", "dishwash", "cleaner",
        "toilet", "tissue", "freshener", "freshner", "floor",
        "bleach", "rin", "wheel", "surf", "soap", "handwash",
        "lizol", "vixol", "domex", "rok", "towel", "sponge"
    }

    personal_tokens = {
        "shampoo", "conditioner", "toothpaste", "toothbrush",
        "cream", "lotion", "facewash", "deodorant", "perfume",
        "lipstick", "powder", "vaseline", "ponds", "dove",
        "hair", "oil", "gel", "face", "wash", "soap"
    }

    non_food_tokens = {
        "toy", "car", "doll", "khata", "pen", "pencil",
        "eraser", "sharpner", "book", "file", "scissors", "clip",
        "umbrella", "card", "sticker", "remote", "battery",
        "charger", "tshirt", "vest", "bra", "orna", "lungi"
    }

    kitchen_tool_tokens = {
        "pan", "knife", "spoon", "bowl", "plate", "cutter",
        "donut", "mould", "mold", "tray", "container", "glass",
        "cup", "mug", "khunti", "karai", "tawa", "pot",
        "pressure", "cooker", "beater", "grater", "peeler",
        "chipper", "strainer"
    }

    if has_any(tokens, meat_tokens):
        return "meat"

    if has_any(tokens, chicken_tokens):
        return "chicken"

    if has_any(tokens, fish_tokens):
        return "fish"

    if has_any(tokens, rice_tokens):
        return "rice"

    if has_any(tokens, spice_tokens):
        return "spice"

    if has_any(tokens, cooking_tokens):
        return "cooking_essential"

    if has_any(tokens, oil_tokens):
        if "hair" in tokens or "coconut" in tokens and "hair" in text:
            return "personal_care"
        return "oil"

    if has_any(tokens, dairy_tokens):
        return "dairy"

    if has_any(tokens, dessert_tokens):
        return "dessert"

    if has_any(tokens, bakery_tokens):
        return "bakery"

    if has_any(tokens, beverage_tokens):
        return "beverage"

    if has_any(tokens, snacks_tokens):
        return "snacks"

    if has_any(tokens, cleaning_tokens):
        return "cleaning"

    if has_any(tokens, personal_tokens):
        return "personal_care"

    if has_any(tokens, kitchen_tool_tokens):
        return "kitchen_tool"

    if has_any(tokens, non_food_tokens):
        return "non_food"

    return "other"


def infer_basket_intent(item_names):
    families = [get_product_family(x) for x in item_names]
    family_set = set(families)
    names_text = normalize_text(" ".join(item_names))

    if "biryani" in names_text or "biriyani" in names_text or "kacchi" in names_text:
        if "beef" in names_text or "mutton" in names_text or "meat" in family_set:
            return "beef_biryani"
        if "chicken" in family_set:
            return "chicken_biryani"
        return "biryani"

    if "meat" in family_set:
        return "beef_cooking"

    if "chicken" in family_set:
        return "chicken_cooking"

    if "fish" in family_set:
        if "hilsha" in names_text or "ilish" in names_text:
            return "hilsha_cooking"
        if "shrimp" in names_text or "bagda" in names_text or "golda" in names_text:
            return "shrimp_cooking"
        return "fish_cooking"

    if "rice" in family_set and family_set.intersection({"cooking_essential", "spice", "oil"}):
        return "daily_meal_cooking"

    if "rice" in family_set:
        return "rice_meal"

    if "dessert" in family_set:
        return "dessert"

    if "bakery" in family_set and family_set.intersection({"dairy", "dessert"}):
        return "breakfast"

    if "bakery" in family_set:
        return "breakfast"

    if family_set.intersection({"snacks", "beverage"}):
        return "snacks"

    if "cleaning" in family_set:
        return "cleaning"

    if "kitchen_tool" in family_set:
        return "kitchen_tool"

    if "personal_care" in family_set:
        return "personal_care"

    return "general"


def rule_boost_for_basket(basket_intent, candidate_name):
    candidate_family = get_product_family(candidate_name)
    candidate_text = normalize_text(candidate_name)

    if basket_intent in {"beef_biryani", "chicken_biryani", "biryani"}:
        if "biryani" in candidate_text or "biriyani" in candidate_text or "kacchi" in candidate_text:
            return 0.85
        if candidate_family in {"spice", "cooking_essential", "oil", "dairy"}:
            return 0.65
        if candidate_family == "beverage" and ("borhani" in candidate_text or "labang" in candidate_text):
            return 0.60
        if candidate_family == "rice":
            return 0.35

    if basket_intent in {"beef_cooking", "chicken_cooking"}:
        if candidate_family in {"cooking_essential", "spice", "oil", "rice"}:
            return 0.60
        if candidate_family == "beverage" and ("borhani" in candidate_text or "labang" in candidate_text):
            return 0.45

    if basket_intent in {"fish_cooking", "hilsha_cooking", "shrimp_cooking"}:
        if basket_intent == "hilsha_cooking" and ("sorshe" in candidate_text or "mustard" in candidate_text):
            return 0.85
        if basket_intent == "shrimp_cooking" and ("coconut" in candidate_text or "milk" in candidate_text):
            return 0.75
        if candidate_family in {"cooking_essential", "spice", "oil", "rice"}:
            return 0.60

    if basket_intent in {"daily_meal_cooking", "rice_meal"}:
        if candidate_family in {"meat", "chicken", "fish", "cooking_essential", "spice", "oil"}:
            return 0.55
        if candidate_family == "beverage" and ("borhani" in candidate_text or "labang" in candidate_text):
            return 0.30

    if basket_intent == "dessert":
        if candidate_family in {"dessert", "dairy"}:
            return 0.65
        if candidate_family == "bakery":
            return 0.35

    if basket_intent == "breakfast":
        if candidate_family in {"dairy", "bakery", "dessert", "beverage"}:
            return 0.50
        if candidate_family in {"snacks"}:
            return 0.25

    if basket_intent == "snacks":
        if candidate_family in {"snacks", "beverage", "bakery"}:
            return 0.50

    if basket_intent == "cleaning":
        if candidate_family == "cleaning":
            return 0.65

    if basket_intent == "kitchen_tool":
        if candidate_family == "kitchen_tool":
            return 0.55
        if candidate_family in {"dessert", "bakery"}:
            return 0.35

    if basket_intent == "personal_care":
        if candidate_family == "personal_care":
            return 0.60

    return 0.0


def hard_block_candidate(basket_intent, candidate_name):
    candidate_family = get_product_family(candidate_name)

    food_intents = {
        "beef_biryani",
        "chicken_biryani",
        "biryani",
        "beef_cooking",
        "chicken_cooking",
        "fish_cooking",
        "hilsha_cooking",
        "shrimp_cooking",
        "daily_meal_cooking",
        "rice_meal",
        "dessert",
        "breakfast",
        "snacks"
    }

    if basket_intent in food_intents:
        if candidate_family in {"cleaning", "personal_care", "non_food", "kitchen_tool"}:
            return True

    if basket_intent == "cleaning":
        if candidate_family not in {"cleaning", "other"}:
            return True

    if basket_intent == "personal_care":
        if candidate_family not in {"personal_care", "other"}:
            return True

    if basket_intent == "kitchen_tool":
        if candidate_family in {"cleaning", "personal_care", "non_food"}:
            return True

    return False


def same_type_penalty(input_families, candidate_name):
    candidate_family = get_product_family(candidate_name)

    no_repeat_families = {
        "meat",
        "chicken",
        "fish",
        "rice",
        "dessert",
        "bakery",
        "beverage",
        "cleaning",
        "personal_care",
        "kitchen_tool"
    }

    if candidate_family in input_families and candidate_family in no_repeat_families:
        return 0.35

    return 0.0