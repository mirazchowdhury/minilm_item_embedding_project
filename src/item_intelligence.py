import re
import pandas as pd
from collections import Counter


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Merged extensive spelling mistakes and variations fix
    replacements = {
        "&": " and ", "+": " plus ", "b2g1": " buy two get one ", "b3g1": " buy three get one ",
        "gm": " g ", "gms": " g ", "kg": " kg ", "ltr": " ltr ", "lt": " ltr ",
        "pcs": " pcs ", "pc": " pcs ", "pkt": " packet ", "powder": " powder",
        "turmaric": "turmeric", "corinder": "coriander", "coriender": "coriander",
        "suger": "sugar", "shemai": "semai", "samai": "semai", "lachcha": "lacha",
        "lascha": "lacha", "vermicili": "vermicelli", "vermicelli": "vermicelli",
        "soyabin": "soyabean", "soya": "soyabean", "lodized": "iodized",
        "musterd": "mustard", "gurlic": "garlic", "chilli": "chili", "chilly": "chili",
        "biriyani": "biryani", "kacchi": "kachchi", "rost": "roast", "roste": "roast",
        "dish wash": "dishwash", "dish washing": "dishwash", "hand wash": "handwash",
        "face wash": "facewash", "toilet cliner": "toilet cleaner", "toilet clen": "toilet cleaner",
        "detegent": "detergent", "detargent": "detergent", "mushur": "musur",
        "mosur": "musur", "masur": "musur", "mug": "moog", "moog": "moog",
        "alachi": "cardamom", "cardmom": "cardamom", "daruchine": "cinnamon",
        "kismis": "raisin", "raising": "raisin", "yogh": "yogurt", "yougurt": "yogurt",
        "doi": "yogurt", "curd": "yogurt", "labang": "lassi", "matha": "lassi",
        "borhani": "borhani", "kazu": "cashew", "kaju": "cashew", "badam": "nut",
        "pesta": "pistachio", "pistachion": "pistachio", "pasta dana": "pistachio",
        "kalo jam": "black plum", "begun": "brinjal", "potol": "parboil",
        "lau": "bottle gourd", "korolla": "bitter gourd", "jhinga": "ridge gourd",
        "chichinga": "snake gourd", "kachurmukhi": "arum", "kacha holud": "raw turmeric",
        "alu": "potato", "aloo": "potato"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove weights and sizes (e.g., 1kg, 500g, 2.5ltr, 10pcs) using Regex
    text = re.sub(r'\b\d+(?:\.\d+)?\s*(kg|g|gm|l|ml|ltr|pc|pcs|pack|pkt|packet)\b', ' ', text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    text = normalize_text(text)

    # Merged Stop words
    stop_words = {
        "g", "kg", "ltr", "lt", "ml", "packet", "pack", "box", "pcs",
        "pc", "new", "big", "small", "medium", "large", "special",
        "premium", "fresh", "pure", "classic", "regular", "all",
        "combo", "buy", "two", "three", "get", "free", "save",
        "red", "green", "yellow", "white", "black", "blue",
        "china", "deshi", "indian", "thai", "korean", "malaysia",
        "org", "pet", "jar", "tin", "poly", "bottle", "loose",
        "full", "half", "mini", "super", "local", "imported", "frozen"
    }

    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1 and not t.isdigit()]
    return set(tokens)


# Strict Taxonomy based on your Dataset (Merged)
CATEGORY_KEYWORDS = {
    "meat_raw": {"beef", "cow", "mutton", "chicken", "duck", "pigeon", "lever", "liver", "stomach", "bone", "boneless",
                 "breast", "kima", "brain", "wings", "drumsticks", "neck", "sausage", "nuggets", "salami"},
    "fish_raw": {"fish", "rui", "hilsha", "ilish", "shrimp", "bagda", "golda", "pabda", "tengra", "shing", "magur",
                 "pangas", "katol", "koral", "boal", "mrigel", "bele", "poa", "koi", "shole", "laitta", "mola",
                 "kachki", "rupchanda", "tilapia", "telapia", "topashi", "ayir", "bata", "kaikka", "chitol", "gulsha",
                 "crab", "surma", "tuna", "salmon"},
    "staple_grains": {"rice", "chal", "chinigura", "aromatic", "basmati", "miniket", "najirshail", "polao", "pulao",
                      "atob", "atap", "katari", "atta", "maida", "flour", "suji", "sagu"},
    "pulse_dal": {"pulse", "dal", "daal", "lentil", "musur", "moog", "mug", "boot", "chola", "maskalai", "dubli",
                  "khesari"},
    "vegetable": {"potato", "tomato", "onion", "garlic", "ginger", "chili", "chilli", "capsicum", "papaya", "pumpkin",
                  "brinjal", "carrot", "cucumber", "cabbage", "cauliflower", "borboti", "bean", "gourd", "parboil",
                  "ladies", "finger", "usta", "kakrol", "jhinga", "sajna", "radish", "mushroom", "corn", "palong",
                  "shak", "lettuce", "coriander", "mint", "pudina", "lemon", "lau", "bottle", "green", "raw",
                  "turmeric"},
    "fruit": {"apple", "mango", "banana", "grape", "orange", "malta", "pomegranate", "naspati", "dragon", "guava",
              "watermelon", "latkon", "amra", "jamrul", "sofeda", "pineapple", "strawberry", "blackberry", "jumbura",
              "kamranga", "litchi", "rambutan", "honeydew", "dates", "khejur"},
    "spice_masala": {"masala", "spice", "turmeric", "cumin", "coriander", "garam", "biryani", "roast", "tehari",
                     "kachchi", "kabab", "bbq", "tandoori", "haleem", "firni", "kheer", "falooda", "mustard", "kasundi",
                     "panchforon", "pepper", "clove", "cinnamon", "cardamom", "elachi", "mace", "nutmeg", "bay", "leaf",
                     "methi", "oregano", "paprika", "mixed", "powder", "curry", "jafran"},
    "oil_ghee": {"oil", "soyabean", "mustard", "sunflower", "olive", "rice", "bran", "ghee", "butter", "margarine"},
    "salt_sugar": {"salt", "sugar", "misri", "molasses", "jaggery", "zerocal", "splenda", "canderel"},
    "dairy": {"milk", "powder", "dano", "diploma", "marks", "nido", "arla", "aarong", "farm", "fresh", "cheese",
              "yogurt", "curd", "doi", "lassi", "labang", "matha", "cream", "paneer", "borhani"},
    "bakery_dessert": {"bread", "bun", "toast", "cake", "muffin", "croissant", "bakerkhani", "sandwich", "biscuit",
                       "cookies", "semai", "vermicelli", "pudding", "jelly", "chocolate", "cocoa", "flour", "baking",
                       "yeast", "vanilla", "essence", "rose", "water", "kewra", "agar", "gelatine", "pitha"},
    "beverage": {"coke", "cola", "sprite", "pepsi", "fanta", "seven", "7up", "mojo", "dew", "juice", "drink", "tea",
                 "coffee", "nescafe", "latte", "soda", "sharbat", "rooh", "afza", "gatorade", "milkshake"},
    "snacks": {"chips", "kurkure", "lays", "pringles", "chanachur", "candy", "lollipop", "gum", "cracker", "nimki",
               "murali", "badam", "nut", "papri", "puffed", "muri", "popcorn", "fuchka", "gol", "gappa", "chatpati",
               "halim", "haleem"},
    "frozen_ready_meal": {"sausage", "samosa", "singara", "spring", "roll", "fries", "paratha", "roti", "chapati",
                          "burger", "patty", "meatball", "kabab", "lollipop", "strips", "popcorn", "dumpling", "momo",
                          "pizza", "wonton"},
    "noodles_pasta": {"noodles", "ramen", "maggi", "mama", "cocola", "doodles", "pasta", "macaroni", "spaghetti",
                      "penne", "fusilli", "stick"},
    "sauce_condiment": {"sauce", "ketchup", "mayonnaise", "mayo", "vinegar", "soya", "soy", "oyster", "bbq", "pickle",
                        "chutney", "dressing", "tabasco"},
    "cleaning": {"vim", "dishwash", "harpic", "detergent", "rin", "wheel", "surf", "excel", "toilet", "cleaner",
                 "floor", "lizol", "domex", "vixol", "rok", "soap", "handwash", "tissue", "towel", "napkin", "sponge",
                 "brush", "bleach", "glass", "phenyl", "naphthalene", "air", "freshener", "odonil"},
    "personal_care": {"shampoo", "conditioner", "toothpaste", "toothbrush", "facewash", "cream", "lotion", "vaseline",
                      "dove", "lux", "dettol", "lifebuoy", "ponds", "sunsilk", "sensodyne", "pepsodent", "colgate",
                      "nivea", "fair", "lovely", "garnier", "parachute", "meril", "savlon", "deodorant", "body",
                      "spray", "perfume"},
    "baby_care": {"baby", "cerelac", "lactogen", "nan", "feeding", "nipple", "diaper", "pampers", "huggies", "kodomo",
                  "johnsons", "mothercare", "pacifier"},
    "stationery_toy": {"pen", "pencil", "eraser", "khata", "notebook", "marker", "glue", "toy", "ball", "card", "doll",
                       "car", "puzzle", "sharpner", "stapler", "book", "colour", "color", "drawing", "board"},
    "kitchen_household": {"plate", "bowl", "spoon", "mug", "glass", "container", "jar", "tawa", "pan", "karai",
                          "cooker", "knife", "scissor", "peeler", "grater", "tray", "lunch", "box", "water", "pot",
                          "jug", "basket", "cover", "foil"}
}


def detect_category(item_name):
    tokens = tokenize(item_name)
    scores = {cat: len(tokens.intersection(keywords)) for cat, keywords in CATEGORY_KEYWORDS.items() if
              len(tokens.intersection(keywords)) > 0}
    return max(scores, key=scores.get) if scores else "general"


def detect_family(item_name):
    tokens = tokenize(item_name)
    text = normalize_text(item_name)

    family_rules = [
        ("beef", {"beef", "cow", "bone", "stomach", "lever"}),
        ("chicken", {"chicken", "cock", "hen"}),
        ("mutton", {"mutton"}),
        ("fish", {"fish", "rui", "hilsha", "shrimp", "pabda", "tengra", "shing", "magur", "pangas", "katol", "koral"}),
        ("rice", {"rice", "chinigura", "basmati", "miniket", "najirshail"}),
        ("atta_maida", {"atta", "maida", "flour"}),
        ("potato", {"potato"}),
        ("tomato", {"tomato"}),
        ("onion", {"onion"}),
        ("garlic", {"garlic"}),
        ("ginger", {"ginger"}),
        ("green_chili", {"chili", "chilli"}),
        ("oil", {"oil", "soyabean", "mustard"}),
        ("salt", {"salt"}),
        ("sugar", {"sugar"}),
        ("milk", {"milk", "dano", "diploma", "nido", "arla"}),
        ("yogurt", {"yogurt", "curd", "doi", "borhani"}),
        ("bread", {"bread", "bun", "toast", "pitha"}),
        ("egg", {"egg"}),
        ("semai", {"semai", "vermicelli", "lacha"}),
        ("tea_coffee", {"tea", "coffee", "nescafe"}),
        ("soft_drink", {"coke", "cola", "sprite", "pepsi", "fanta", "7up", "mojo"}),
        ("chips", {"chips", "kurkure", "lays", "pringles"}),
        ("cleaning", {"vim", "harpic", "detergent", "dishwash", "toilet", "cleaner", "wheel", "rin", "surf"}),
        ("tissue", {"tissue", "napkin", "towel"}),
        ("noodles", {"noodles", "ramen", "maggi"}),
        ("sauce", {"sauce", "ketchup", "mayonnaise", "vinegar"}),
        ("fruit", {"apple", "mango", "banana", "grape", "orange", "malta"})
    ]

    for family, words in family_rules:
        if tokens.intersection(words):
            return family

    if "masala" in text:
        return "masala"

    return detect_category(item_name)


def enrich_item_catalog(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    if "item_name" not in df.columns:
        raise ValueError("CSV file must contain item_name column")

    df["item_name"] = df["item_name"].astype(str)
    df["normalized_item_name"] = df["item_name"].apply(normalize_text)
    df["tokens"] = df["item_name"].apply(lambda x: " ".join(sorted(tokenize(x))))
    df["category"] = df["item_name"].apply(detect_category)
    df["family"] = df["item_name"].apply(detect_family)

    df = df.drop_duplicates(subset=["normalized_item_name"]).reset_index(drop=True)
    df["item_index"] = range(len(df))

    df.to_csv(output_csv_path, index=False)
    print(f"Saved enriched catalog: {output_csv_path}")
    print(f"Total unique items: {len(df)}")
    return df