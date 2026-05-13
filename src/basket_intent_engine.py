from collections import Counter

from item_intelligence import tokenize, detect_category, detect_family


INTENT_PROFILES = {
    "beef_cooking": {
        "required_families": {
            "onion", "garlic", "ginger", "green_chili", "oil", "salt",
            "rice", "potato", "masala"
        },
        "good_categories": {
            "vegetable", "spice_masala", "oil_ghee", "rice", "salt_sugar",
            "dairy", "beverage"
        },
        "bad_categories": {
            "cleaning", "personal_care", "baby_care", "stationery_toy",
            "dessert", "snacks"
        },
    },

    "chicken_cooking": {
        "required_families": {
            "onion", "garlic", "ginger", "green_chili", "oil", "salt",
            "rice", "potato", "egg", "masala"
        },
        "good_categories": {
            "vegetable", "spice_masala", "oil_ghee", "rice", "salt_sugar",
            "dairy", "beverage", "frozen_ready_meal"
        },
        "bad_categories": {
            "cleaning", "personal_care", "baby_care", "stationery_toy",
            "dessert"
        },
    },

    "fish_cooking": {
        "required_families": {
            "onion", "garlic", "ginger", "green_chili", "oil", "salt",
            "rice", "masala"
        },
        "good_categories": {
            "vegetable", "spice_masala", "oil_ghee", "rice", "salt_sugar"
        },
        "bad_categories": {
            "cleaning", "personal_care", "baby_care", "stationery_toy",
            "dessert", "snacks"
        },
    },

    "biryani_cooking": {
        "required_families": {
            "rice", "onion", "garlic", "ginger", "green_chili", "oil",
            "salt", "masala", "yogurt", "borhani"
        },
        "good_categories": {
            "meat_raw", "rice", "vegetable", "spice_masala", "oil_ghee",
            "salt_sugar", "dairy", "beverage"
        },
        "bad_categories": {
            "cleaning", "personal_care", "baby_care", "stationery_toy",
            "dessert"
        },
    },

    "daily_meal_cooking": {
        "required_families": {
            "onion", "garlic", "ginger", "green_chili", "oil", "salt",
            "rice", "fish", "chicken", "beef", "masala"
        },
        "good_categories": {
            "meat_raw", "fish_raw", "rice", "pulse_dal", "vegetable",
            "spice_masala", "oil_ghee", "salt_sugar"
        },
        "bad_categories": {
            "cleaning", "personal_care", "baby_care", "stationery_toy",
            "dessert", "snacks", "beverage"
        },
    },

    "dessert": {
        "required_families": {
            "sugar", "milk", "semai", "suji", "sagu", "ghee"
        },
        "good_categories": {
            "dairy", "dessert", "salt_sugar", "bakery", "fruit"
        },
        "bad_categories": {
            "meat_raw", "fish_raw", "vegetable", "spice_masala",
            "cleaning", "personal_care", "stationery_toy"
        },
    },

    "breakfast": {
        "required_families": {
            "bread", "egg", "milk", "tea", "coffee", "butter"
        },
        "good_categories": {
            "bakery", "dairy", "beverage", "fruit", "dessert",
            "frozen_ready_meal"
        },
        "bad_categories": {
            "fish_raw", "meat_raw", "cleaning", "personal_care",
            "stationery_toy"
        },
    },

    "snacks_party": {
        "required_families": {
            "soft_drink", "chips", "sauce", "chicken"
        },
        "good_categories": {
            "beverage", "snacks", "frozen_ready_meal", "sauce_condiment",
            "bakery", "dessert"
        },
        "bad_categories": {
            "fish_raw", "rice", "pulse_dal", "cleaning", "personal_care"
        },
    },

    "cleaning": {
        "required_families": {
            "cleaning", "tissue"
        },
        "good_categories": {
            "cleaning", "kitchen_household"
        },
        "bad_categories": {
            "meat_raw", "fish_raw", "rice", "pulse_dal", "vegetable",
            "fruit", "dairy", "bakery", "dessert", "snacks"
        },
    },
}


def detect_basket_intent(item_names):
    categories = []
    families = []
    all_tokens = set()

    for name in item_names:
        categories.append(detect_category(name))
        families.append(detect_family(name))
        all_tokens.update(tokenize(name))

    category_count = Counter(categories)
    family_set = set(families)

    has_beef = "beef" in family_set
    has_chicken = "chicken" in family_set
    has_mutton = "mutton" in family_set
    has_fish = "fish" in family_set
    has_rice = "rice" in family_set
    has_potato = "potato" in family_set
    has_onion = "onion" in family_set
    has_milk = "milk" in family_set
    has_sugar = "sugar" in family_set
    has_bread = "bread" in family_set
    has_egg = "egg" in family_set
    has_semai = "semai" in family_set
    has_suji = "suji" in family_set
    has_soft_drink = "soft_drink" in family_set
    has_chips = "chips" in family_set

    if category_count["cleaning"] >= 2:
        return "cleaning"

    if has_rice and ("biryani" in all_tokens or "kachchi" in all_tokens):
        return "biryani_cooking"

    if has_rice and (has_beef or has_chicken or has_mutton) and has_onion:
        return "biryani_cooking"

    if has_beef or has_mutton:
        return "beef_cooking"

    if has_chicken:
        return "chicken_cooking"

    if has_fish:
        return "fish_cooking"

    if has_rice and (has_potato or has_onion):
        return "daily_meal_cooking"

    if has_rice:
        return "daily_meal_cooking"

    if has_semai or has_suji or (has_milk and has_sugar):
        return "dessert"

    if has_bread and (has_egg or has_milk):
        return "breakfast"

    if has_soft_drink and has_chips:
        return "snacks_party"

    if category_count["snacks"] >= 2 or category_count["beverage"] >= 2:
        return "snacks_party"

    return "daily_meal_cooking"


def get_intent_profile(intent):
    return INTENT_PROFILES.get(intent, INTENT_PROFILES["daily_meal_cooking"])