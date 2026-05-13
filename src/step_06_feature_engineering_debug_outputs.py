import re
import ast
import difflib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. Project path setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "debug_feature_outputs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Debug output settings
# =========================================================

# SAVE_FULL_MATRIX_OUTPUTS = True
# SAVE_READABLE_OUTPUTS = True
# SAVE_EMBEDDING_OUTPUTS = True

SAVE_FULL_MATRIX_OUTPUTS = False
SAVE_READABLE_OUTPUTS = True
SAVE_EMBEDDING_OUTPUTS = True

MATRIX_CHUNK_SIZE = 250

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 3000)


# =========================================================
# 3. Data file auto detection
# =========================================================

possible_files = [
    DATA_INPUT_DIR / "customer_purchase_with_basket_list.csv",
    DATA_INPUT_DIR / "form_paste_clean_v2_baskets_list .csv",
    DATA_INPUT_DIR / "form_paste_clean_v2.csv",
    DATA_INPUT_DIR / "form_paste_clean_v2 .csv",
    DATA_INPUT_DIR / "form_paste.csv",
    DATA_INPUT_DIR / "main_data.csv",
    DATA_INPUT_DIR / "transactions.csv",
    DATA_DIR / "customer_purchase_with_basket_list.csv",
    DATA_DIR / "form_paste_clean_v2_baskets_list .csv",
    DATA_DIR / "form_paste_clean_v2.csv",
    DATA_DIR / "form_paste_clean_v2 .csv",
    DATA_DIR / "form_paste.csv",
    DATA_DIR / "main_data.csv",
    DATA_DIR / "transactions.csv",
]

DATA_FILE = None

for file in possible_files:
    if file.exists():
        DATA_FILE = file
        break

if DATA_FILE is None:
    print("\nAvailable files inside data folder:")
    for file in DATA_DIR.rglob("*.csv"):
        print(file)

    raise FileNotFoundError(
        "No valid data file found. Please put your CSV file inside data/input folder."
    )

raw_df = pd.read_csv(DATA_FILE)

print("\nLoaded data file:")
print(DATA_FILE)

print("\nColumns:")
print(raw_df.columns.tolist())


# =========================================================
# 4. Flexible column finder
# =========================================================

def find_col(possible_names, dataframe):
    lower_map = {c.lower().strip(): c for c in dataframe.columns}

    for name in possible_names:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]

    for c in dataframe.columns:
        c_low = c.lower().strip()
        for name in possible_names:
            if name.lower().strip() in c_low:
                return c

    return None


transaction_col = find_col([
    "transactionId",
    "transaction_id",
    "transactionid",
    "invoice",
    "invoiceNo",
    "orderId",
    "order_id",
    "basketId",
    "basket_id",
    "purchaseId",
    "purchase_id"
], raw_df)

customer_col = find_col([
    "customerId",
    "customer_id",
    "customerid",
    "userId",
    "user_id",
    "clientId",
    "client_id"
], raw_df)

item_col = find_col([
    "item_name",
    "itemName",
    "itemname",
    "product_name",
    "productName",
    "product",
    "items",
    "itemNameSet",
    "item_name_set",
    "name"
], raw_df)

category_col = find_col([
    "category",
    "categoryName",
    "category_name",
    "categorySet",
    "category_set"
], raw_df)

date_col = find_col([
    "orderDate",
    "order_date",
    "purchaseDate",
    "purchase_date",
    "date"
], raw_df)

if customer_col is None:
    raise ValueError("Customer column not found.")

if item_col is None:
    raise ValueError("Item name column not found.")

if transaction_col is None:
    if date_col is not None:
        raw_df["generated_basket_id"] = (
            raw_df[customer_col].astype(str).str.strip()
            + "_"
            + raw_df[date_col].astype(str).str.strip()
        )
        transaction_col = "generated_basket_id"
    else:
        raw_df["generated_basket_id"] = raw_df.index.astype(str)
        transaction_col = "generated_basket_id"

print("\nDetected columns:")
print("Transaction column:", transaction_col)
print("Customer column:", customer_col)
print("Item column:", item_col)
print("Category column:", category_col)
print("Date column:", date_col)


# =========================================================
# 5. List parser and basket explode
# =========================================================

def parse_possible_list(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, set, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, set, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if "||" in text:
        return [x.strip() for x in text.split("||") if x.strip()]

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]

    return [text]


def build_long_dataframe(input_df):
    needed_cols = [transaction_col, customer_col, item_col]

    if category_col is not None:
        needed_cols.append(category_col)

    temp = input_df[needed_cols].copy()

    temp[transaction_col] = temp[transaction_col].astype(str).str.strip()
    temp[customer_col] = temp[customer_col].astype(str).str.strip()

    long_rows = []

    for _, row in temp.iterrows():
        trx = str(row[transaction_col]).strip()
        customer = str(row[customer_col]).strip()

        item_values = parse_possible_list(row[item_col])

        if category_col is not None:
            category_values = parse_possible_list(row[category_col])
        else:
            category_values = []

        for pos, item_name in enumerate(item_values):
            category_value = ""

            if len(category_values) == len(item_values):
                category_value = category_values[pos]
            elif len(category_values) == 1:
                category_value = category_values[0]

            item_name = str(item_name).strip()

            if item_name == "" or item_name.lower() == "nan":
                continue

            long_rows.append({
                "transaction_id": trx,
                "customer_id": customer,
                "item_name": item_name,
                "category": str(category_value).strip()
            })

    output_df = pd.DataFrame(long_rows)

    if output_df.empty:
        raise ValueError("No item rows found after exploding basket data.")

    output_df = output_df.dropna(subset=["transaction_id", "customer_id", "item_name"])

    output_df["transaction_id"] = output_df["transaction_id"].astype(str).str.strip()
    output_df["customer_id"] = output_df["customer_id"].astype(str).str.strip()
    output_df["item_name"] = output_df["item_name"].astype(str).str.strip()
    output_df["category"] = output_df["category"].astype(str).str.strip()

    output_df = output_df[output_df["item_name"] != ""]
    output_df = output_df.drop_duplicates(
        subset=["transaction_id", "customer_id", "item_name"]
    ).reset_index(drop=True)

    return output_df


df = build_long_dataframe(raw_df)

transaction_col = "transaction_id"
customer_col = "customer_id"
item_col = "item_name"
category_col = "category"

df.to_csv(
    OUTPUT_DIR / "00_clean_long_transaction_data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nClean long data sample:")
print(df.head(20))

print("\nClean long data saved:")
print(OUTPUT_DIR / "00_clean_long_transaction_data.csv")

print("\nUsing normalized long data columns:")
print("Transaction column:", transaction_col)
print("Customer column:", customer_col)
print("Item column:", item_col)
print("Category column:", category_col)


# =========================================================
# 6. Item, customer, transaction indexing
# =========================================================

items = sorted(df[item_col].unique())
customers = sorted(df[customer_col].unique())
transactions = sorted(df[transaction_col].unique())

item_to_idx = {item: idx for idx, item in enumerate(items)}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

customer_to_idx = {customer: idx for idx, customer in enumerate(customers)}
transaction_to_idx = {trx: idx for idx, trx in enumerate(transactions)}

item_index_df = pd.DataFrame({
    "item_index": list(range(len(items))),
    "item_name": items
})

item_index_df.to_csv(
    MODEL_DIR / "item_index.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved item index mapping:")
print(MODEL_DIR / "item_index.csv")

print("\nBasic counts:")
print("Total rows:", len(df))
print("Unique transactions:", len(transactions))
print("Unique customers:", len(customers))
print("Unique items:", len(items))


# =========================================================
# 7. Save full sparse matrix as chunked CSV
# =========================================================

def save_sparse_matrix_full_csv(matrix, row_labels, col_labels, output_path, chunk_size=250):
    output_path = Path(output_path)

    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path.with_name(
                output_path.stem + "_" + timestamp + output_path.suffix
            )
            print("\nOld CSV file is open in another program.")
            print("Saving new file instead:")
            print(output_path)

    total_rows = matrix.shape[0]

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)

        chunk = matrix[start:end].toarray()

        chunk_df = pd.DataFrame(
            chunk,
            index=row_labels[start:end],
            columns=col_labels
        )

        chunk_df.index.name = "row_label"

        if start == 0:
            chunk_df.to_csv(output_path, mode="w", encoding="utf-8-sig")
        else:
            chunk_df.to_csv(output_path, mode="a", header=False, encoding="utf-8-sig")

        print(f"Saved rows {start} to {end} in {output_path.name}")


# =========================================================
# 8. One hot basket matrix
# =========================================================

row_idx = df["transaction_id"].map(transaction_to_idx).values
col_idx = df["item_name"].map(item_to_idx).values
values = np.ones(len(df), dtype=np.float32)

one_hot_matrix = sparse.csr_matrix(
    (values, (row_idx, col_idx)),
    shape=(len(transactions), len(items))
)

one_hot_matrix.data[:] = 1

sparse.save_npz(MODEL_DIR / "debug_basket_one_hot_matrix.npz", one_hot_matrix)

print("\nOne hot matrix shape:")
print(one_hot_matrix.shape)

sample_one_hot = pd.DataFrame(
    one_hot_matrix[:10, :min(50, len(items))].toarray(),
    index=transactions[:10],
    columns=items[:min(50, len(items))]
)

sample_one_hot.to_csv(
    OUTPUT_DIR / "01_one_hot_matrix_sample.csv",
    encoding="utf-8-sig"
)

print("\nOne hot matrix sample:")
print(sample_one_hot)

if SAVE_FULL_MATRIX_OUTPUTS:
    save_sparse_matrix_full_csv(
        one_hot_matrix,
        transactions,
        items,
        OUTPUT_DIR / "01_one_hot_matrix_FULL.csv",
        chunk_size=MATRIX_CHUNK_SIZE
    )


# =========================================================
# 9. Co occurrence matrix
# =========================================================

co_matrix = one_hot_matrix.T @ one_hot_matrix
co_matrix.setdiag(0)
co_matrix.eliminate_zeros()
co_matrix = co_matrix.tocsr()

sparse.save_npz(MODEL_DIR / "debug_co_occurrence_matrix.npz", co_matrix)

print("\nCo occurrence matrix shape:")
print(co_matrix.shape)

sample_co = pd.DataFrame(
    co_matrix[:min(50, len(items)), :min(50, len(items))].toarray(),
    index=items[:min(50, len(items))],
    columns=items[:min(50, len(items))]
)

sample_co.to_csv(
    OUTPUT_DIR / "02_co_occurrence_matrix_sample.csv",
    encoding="utf-8-sig"
)

print("\nCo occurrence matrix sample:")
print(sample_co)

if SAVE_FULL_MATRIX_OUTPUTS:
    save_sparse_matrix_full_csv(
        co_matrix,
        items,
        items,
        OUTPUT_DIR / "02_co_occurrence_matrix_FULL.csv",
        chunk_size=MATRIX_CHUNK_SIZE
    )


# =========================================================
# 10. Readable co occurrence pairs
# =========================================================

if SAVE_READABLE_OUTPUTS:
    coo = co_matrix.tocoo()

    pair_rows = []

    for i, j, value in zip(coo.row, coo.col, coo.data):
        if i != j and value > 0:
            pair_rows.append({
                "item_a": items[i],
                "item_b": items[j],
                "co_occurrence_count": float(value)
            })

    co_pair_df = pd.DataFrame(pair_rows)

    if len(co_pair_df) > 0:
        co_pair_df = co_pair_df.sort_values(
            by="co_occurrence_count",
            ascending=False
        )

    co_pair_path = OUTPUT_DIR / "02_co_occurrence_READABLE_pairs_FULL.csv"
    co_pair_df.to_csv(co_pair_path, index=False, encoding="utf-8-sig")

    print("\nReadable co occurrence pair output saved:")
    print(co_pair_path)

    print("\nReadable co occurrence pair sample:")
    print(co_pair_df.head(50))


# =========================================================
# 11. Item user matrix
# =========================================================

user_row_idx = df["item_name"].map(item_to_idx).values
user_col_idx = df["customer_id"].map(customer_to_idx).values
user_values = np.ones(len(df), dtype=np.float32)

item_user_matrix = sparse.csr_matrix(
    (user_values, (user_row_idx, user_col_idx)),
    shape=(len(items), len(customers))
)

item_user_matrix.data[:] = 1

sparse.save_npz(MODEL_DIR / "debug_item_user_matrix.npz", item_user_matrix)

print("\nItem user matrix shape:")
print(item_user_matrix.shape)

sample_item_user = pd.DataFrame(
    item_user_matrix[:min(50, len(items)), :min(50, len(customers))].toarray(),
    index=items[:min(50, len(items))],
    columns=customers[:min(50, len(customers))]
)

sample_item_user.to_csv(
    OUTPUT_DIR / "03_item_user_matrix_sample.csv",
    encoding="utf-8-sig"
)

print("\nItem user matrix sample:")
print(sample_item_user)

if SAVE_FULL_MATRIX_OUTPUTS:
    save_sparse_matrix_full_csv(
        item_user_matrix,
        items,
        customers,
        OUTPUT_DIR / "03_item_user_matrix_FULL.csv",
        chunk_size=MATRIX_CHUNK_SIZE
    )


# =========================================================
# 12. Readable item user output
# =========================================================

if SAVE_READABLE_OUTPUTS:
    iu_coo = item_user_matrix.tocoo()

    item_user_rows = []

    for i, j, value in zip(iu_coo.row, iu_coo.col, iu_coo.data):
        if value > 0:
            item_user_rows.append({
                "item_name": items[i],
                "customer_id": customers[j],
                "purchase_binary_value": float(value)
            })

    item_user_readable_df = pd.DataFrame(item_user_rows)

    item_user_readable_path = OUTPUT_DIR / "03_item_user_READABLE_FULL.csv"
    item_user_readable_df.to_csv(
        item_user_readable_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nReadable item user output saved:")
    print(item_user_readable_path)

    print("\nReadable item user sample:")
    print(item_user_readable_df.head(50))


# =========================================================
# 13. Main item embedding from item user matrix using SVD
# =========================================================

svd_dim = 64
svd_dim_user = min(svd_dim, min(item_user_matrix.shape) - 1)

if svd_dim_user < 1:
    raise ValueError("Not enough rows or columns for item user SVD.")

item_user_svd = TruncatedSVD(
    n_components=svd_dim_user,
    random_state=42
)

main_item_embedding = item_user_svd.fit_transform(item_user_matrix)

main_embedding_df = pd.DataFrame(main_item_embedding)
main_embedding_df.insert(0, "item_name", items)

main_embedding_df.head(50).to_csv(
    OUTPUT_DIR / "04_main_item_embedding_from_item_user_svd_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

if SAVE_EMBEDDING_OUTPUTS:
    main_embedding_df.to_csv(
        OUTPUT_DIR / "04_main_item_embedding_from_item_user_svd_FULL.csv",
        index=False,
        encoding="utf-8-sig"
    )

np.save(MODEL_DIR / "debug_main_item_user_svd_embedding.npy", main_item_embedding)

print("\nMain item embedding shape:")
print(main_item_embedding.shape)

print("\nMain item embedding explained variance:")
print(round(float(item_user_svd.explained_variance_ratio_.sum()), 4))

print("\nMain item embedding sample:")
print(main_embedding_df.head(20))


# =========================================================
# 14. Backup item embedding from co occurrence matrix using SVD
# =========================================================

svd_dim_co = min(svd_dim, min(co_matrix.shape) - 1)

if svd_dim_co < 1:
    raise ValueError("Not enough rows or columns for co occurrence SVD.")

co_svd = TruncatedSVD(
    n_components=svd_dim_co,
    random_state=42
)

backup_item_embedding = co_svd.fit_transform(co_matrix)

backup_embedding_df = pd.DataFrame(backup_item_embedding)
backup_embedding_df.insert(0, "item_name", items)

backup_embedding_df.head(50).to_csv(
    OUTPUT_DIR / "05_backup_item_embedding_from_co_occurrence_svd_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

if SAVE_EMBEDDING_OUTPUTS:
    backup_embedding_df.to_csv(
        OUTPUT_DIR / "05_backup_item_embedding_from_co_occurrence_svd_FULL.csv",
        index=False,
        encoding="utf-8-sig"
    )

np.save(MODEL_DIR / "debug_backup_co_occurrence_svd_embedding.npy", backup_item_embedding)

print("\nBackup item embedding shape:")
print(backup_item_embedding.shape)

print("\nBackup item embedding explained variance:")
print(round(float(co_svd.explained_variance_ratio_.sum()), 4))

print("\nBackup item embedding sample:")
print(backup_embedding_df.head(20))


# =========================================================
# 15. MiniLM product text embedding
# =========================================================

def clean_product_text(name):
    text = str(name).lower()
    text = re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


product_texts = [clean_product_text(x) for x in items]

text_embedding_path = MODEL_DIR / "debug_minilm_text_embedding.npy"

if text_embedding_path.exists():
    text_embedding = np.load(text_embedding_path)
    print("\nLoaded MiniLM text embedding from file.")
else:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        text_embedding = model.encode(
            product_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        np.save(text_embedding_path, text_embedding)

        print("\nMiniLM text embedding created and saved.")

    except Exception as e:
        print("\nMiniLM embedding could not be created.")
        print("Reason:", e)
        print("Using zero text embedding only for debug.")

        text_embedding = np.zeros((len(items), 384), dtype=np.float32)

if text_embedding.shape[0] != len(items):
    print("\nWarning: Saved MiniLM embedding item count does not match current item count.")
    print("Rebuilding zero text embedding for safe debug.")
    text_embedding = np.zeros((len(items), 384), dtype=np.float32)

text_embedding_df = pd.DataFrame(text_embedding)
text_embedding_df.insert(0, "item_name", items)

text_embedding_df.head(50).to_csv(
    OUTPUT_DIR / "06_minilm_text_embedding_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

if SAVE_EMBEDDING_OUTPUTS:
    text_embedding_df.to_csv(
        OUTPUT_DIR / "06_minilm_text_embedding_FULL.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\nMiniLM text embedding shape:")
print(text_embedding.shape)

print("\nMiniLM text embedding sample:")
print(text_embedding_df.head(20))


# =========================================================
# 16. Cosine similarity helper
# =========================================================

def safe_cosine_vector(matrix, query_index):
    query_vector = matrix[query_index].reshape(1, -1)
    scores = cosine_similarity(query_vector, matrix)[0]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


# =========================================================
# 17. Product family and intent logic
# =========================================================

def tokenize_name(name):
    text = str(name).lower()
    text = re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", text)
    return set(text.split())


def get_product_family(name):
    tokens = tokenize_name(name)

    if tokens & {"beef", "cow", "mutton", "meat"}:
        return "meat"

    if tokens & {"chicken", "cock", "broiler"}:
        return "chicken"

    if tokens & {
        "fish", "rui", "hilsha", "shrimp", "pabda", "tengra",
        "katol", "boal", "pangas", "koi", "magur", "shing",
        "mrigel", "tilapia", "telapia", "golda", "bagda"
    }:
        return "fish"

    if tokens & {"rice", "chinigura", "basmati", "miniket", "najirshail", "polao"}:
        return "rice"

    if tokens & {"onion", "garlic", "ginger", "chilli", "chili", "tomato", "potato", "coriander", "lemon"}:
        return "cooking_essential"

    if tokens & {
        "masala", "cumin", "turmeric", "coriander", "pepper",
        "cinnamon", "cardmom", "cardamom", "clove", "panchforon",
        "methi", "bay", "leaf", "powder"
    }:
        return "spice"

    if tokens & {"oil", "soyabean", "soyabin", "mustard", "ghee"}:
        return "oil"

    if tokens & {"milk", "butter", "cheese", "yogurt", "curd", "doi", "cream", "paneer"}:
        return "dairy"

    if tokens & {
        "sugar", "semai", "samai", "suji", "sagu", "vermicelli",
        "custard", "falooda", "firni", "kheer", "jelly", "raisin",
        "kismis", "dates"
    }:
        return "dessert"

    if tokens & {"bread", "toast", "bun", "cake", "biscuit", "cookies", "cracker"}:
        return "bakery"

    if tokens & {"coke", "cola", "sprite", "pepsi", "juice", "borhani", "labang", "matha", "fanta", "dew", "7up"}:
        return "beverage"

    if tokens & {"chips", "chanachur", "kurkure", "lays", "pringles", "snacks", "nuggets", "sausage", "fries"}:
        return "snacks"

    if tokens & {
        "vim", "harpic", "detergent", "dishwash", "cleaner",
        "toilet", "tissue", "freshener", "freshner", "floor",
        "bleach", "rin", "wheel", "surf", "soap"
    }:
        return "cleaning"

    if tokens & {
        "shampoo", "conditioner", "toothpaste", "toothbrush",
        "cream", "lotion", "facewash", "deodorant", "perfume",
        "lipstick", "powder"
    }:
        return "personal_care"

    if tokens & {
        "toy", "car", "doll", "khata", "pen", "pencil", "eraser",
        "sharpner", "book", "file", "scissors", "clip"
    }:
        return "non_food"

    if tokens & {
        "pan", "knife", "spoon", "bowl", "plate", "cutter",
        "donut", "mould", "mold", "tray", "container", "glass",
        "cup", "mug", "khunti", "karai", "tawa"
    }:
        return "kitchen_tool"

    return "other"


def infer_product_intent(name):
    family = get_product_family(name)

    if family in {"meat", "chicken"}:
        return "meat_cooking"

    if family == "fish":
        return "fish_cooking"

    if family in {"rice", "cooking_essential", "spice", "oil"}:
        return "daily_meal_cooking"

    if family in {"dessert", "dairy"}:
        return "dessert_or_breakfast"

    if family == "bakery":
        return "breakfast"

    if family in {"beverage", "snacks"}:
        return "snacks"

    if family == "cleaning":
        return "cleaning"

    if family == "kitchen_tool":
        return "kitchen_tool"

    if family == "personal_care":
        return "personal_care"

    return "general"


def same_type_penalty(input_item, candidate_item):
    input_family = get_product_family(input_item)
    candidate_family = get_product_family(candidate_item)

    if input_family == candidate_family and input_family in {
        "meat", "chicken", "fish", "rice", "dessert",
        "bakery", "beverage", "cleaning", "personal_care"
    }:
        return 0.35

    return 0.0


def rule_boost(input_item, candidate_item):
    input_family = get_product_family(input_item)
    candidate_family = get_product_family(candidate_item)

    cooking_companion = {
        "cooking_essential",
        "spice",
        "oil",
        "rice",
        "beverage"
    }

    fish_companion = {
        "cooking_essential",
        "spice",
        "oil",
        "rice"
    }

    dessert_companion = {
        "dessert",
        "dairy",
        "bakery"
    }

    breakfast_companion = {
        "dairy",
        "bakery",
        "dessert",
        "beverage"
    }

    snacks_companion = {
        "snacks",
        "beverage",
        "bakery"
    }

    cleaning_companion = {
        "cleaning"
    }

    kitchen_tool_companion = {
        "kitchen_tool",
        "bakery",
        "dessert"
    }

    if input_family in {"meat", "chicken"} and candidate_family in cooking_companion:
        return 0.30

    if input_family == "fish" and candidate_family in fish_companion:
        return 0.30

    if input_family == "rice" and candidate_family in {"meat", "chicken", "fish", "cooking_essential", "spice", "oil", "beverage"}:
        return 0.25

    if input_family == "dessert" and candidate_family in dessert_companion:
        return 0.30

    if input_family == "bakery" and candidate_family in breakfast_companion:
        return 0.25

    if input_family in {"beverage", "snacks"} and candidate_family in snacks_companion:
        return 0.25

    if input_family == "cleaning" and candidate_family in cleaning_companion:
        return 0.30

    if input_family == "kitchen_tool" and candidate_family in kitchen_tool_companion:
        return 0.20

    return 0.0


def hard_block_candidate(input_item, candidate_item):
    input_family = get_product_family(input_item)
    candidate_family = get_product_family(candidate_item)

    food_families = {
        "meat", "chicken", "fish", "rice", "cooking_essential",
        "spice", "oil", "dairy", "dessert", "bakery",
        "beverage", "snacks"
    }

    if input_family in food_families and candidate_family in {"cleaning", "personal_care", "non_food"}:
        return True

    if input_family in {"cleaning", "personal_care"} and candidate_family in food_families:
        return True

    if input_family == "kitchen_tool" and candidate_family in {"personal_care", "cleaning"}:
        return True

    return False


# =========================================================
# 18. Input item resolver
# =========================================================

def normalize_for_match(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


normalized_item_map = {normalize_for_match(item): item for item in items}


def resolve_item_name(user_text):
    user_text = str(user_text).strip()

    if user_text in item_to_idx:
        return user_text

    normalized = normalize_for_match(user_text)

    if normalized in normalized_item_map:
        return normalized_item_map[normalized]

    candidates = difflib.get_close_matches(
        normalized,
        list(normalized_item_map.keys()),
        n=10,
        cutoff=0.45
    )

    if len(candidates) > 0:
        print("\nInput item exact match hoy nai.")
        print("Closest matches:")

        for idx, cand in enumerate(candidates, start=1):
            print(f"{idx}. {normalized_item_map[cand]}")

        selected = normalized_item_map[candidates[0]]

        print("\nAuto selected closest item:")
        print(selected)

        return selected

    possible_matches = [
        item for item in items
        if normalized in normalize_for_match(item) or normalize_for_match(item) in normalized
    ]

    if len(possible_matches) > 0:
        print("\nInput item exact match hoy nai.")
        print("Possible substring matches:")

        for idx, item in enumerate(possible_matches[:20], start=1):
            print(f"{idx}. {item}")

        selected = possible_matches[0]

        print("\nAuto selected possible item:")
        print(selected)

        return selected

    return None


# =========================================================
# 19. Recommendation debug function
# =========================================================

def debug_single_item(input_item, top_n=20):
    resolved_item = resolve_item_name(input_item)

    if resolved_item is None:
        print("\nInput item not found:")
        print(input_item)
        return None

    input_item = resolved_item
    q_idx = item_to_idx[input_item]

    pair_counts = co_matrix[q_idx].toarray().ravel()

    item_user_sim = safe_cosine_vector(main_item_embedding, q_idx)
    co_sim = safe_cosine_vector(backup_item_embedding, q_idx)
    text_sim = safe_cosine_vector(text_embedding, q_idx)

    max_pair_count = max(float(pair_counts.max()), 1.0)

    rows = []

    for i, candidate in enumerate(items):
        if i == q_idx:
            continue

        pair_count = float(pair_counts[i])

        pair_score = np.log1p(pair_count)
        pair_score = pair_score / np.log1p(max_pair_count)

        raw_score = (
            0.35 * float(item_user_sim[i])
            + 0.30 * float(co_sim[i])
            + 0.20 * float(text_sim[i])
            + 0.15 * float(pair_score)
        )

        boost = rule_boost(input_item, candidate)
        penalty = same_type_penalty(input_item, candidate)
        blocked = hard_block_candidate(input_item, candidate)

        score_after_rule = raw_score + boost
        final_score_after_penalty = score_after_rule * (1.0 - penalty)

        if blocked:
            final_score_after_penalty = 0.0

        rows.append({
            "input_item": input_item,
            "input_family": get_product_family(input_item),
            "input_intent": infer_product_intent(input_item),
            "candidate_item": candidate,
            "candidate_family": get_product_family(candidate),
            "candidate_intent": infer_product_intent(candidate),
            "pair_count": pair_count,
            "pair_score": round(float(pair_score), 4),
            "item_user_cosine_similarity": round(float(item_user_sim[i]), 4),
            "co_occurrence_cosine_similarity": round(float(co_sim[i]), 4),
            "minilm_text_cosine_similarity": round(float(text_sim[i]), 4),
            "combined_score_before_rule": round(float(raw_score), 4),
            "business_rule_boost": round(float(boost), 4),
            "combined_score_after_rule": round(float(score_after_rule), 4),
            "same_type_penalty": round(float(penalty), 4),
            "hard_blocked": blocked,
            "final_score_after_penalty": round(float(final_score_after_penalty), 4),
        })

    result = pd.DataFrame(rows)

    result_before_rule = result.sort_values(
        "combined_score_before_rule",
        ascending=False
    )

    result_after_rule = result.sort_values(
        "combined_score_after_rule",
        ascending=False
    )

    result_after_penalty = result.sort_values(
        "final_score_after_penalty",
        ascending=False
    )

    result_before_rule.to_csv(
        OUTPUT_DIR / "07_recommendation_before_rule_FULL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_after_rule.to_csv(
        OUTPUT_DIR / "08_recommendation_after_rule_FULL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_after_penalty.to_csv(
        OUTPUT_DIR / "09_recommendation_after_penalty_FULL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_before_rule.head(top_n).to_csv(
        OUTPUT_DIR / "07_recommendation_before_rule_TOP.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_after_rule.head(top_n).to_csv(
        OUTPUT_DIR / "08_recommendation_after_rule_TOP.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_after_penalty.head(top_n).to_csv(
        OUTPUT_DIR / "09_recommendation_after_penalty_TOP.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nInput item:")
    print(input_item)

    print("\nInput product family:")
    print(get_product_family(input_item))

    print("\nInput intent:")
    print(infer_product_intent(input_item))

    print("\nTop result before business rule:")
    print(result_before_rule.head(top_n))

    print("\nTop result after business rule:")
    print(result_after_rule.head(top_n))

    print("\nTop result after same type penalty and hard block:")
    print(result_after_penalty.head(top_n))

    return result_after_penalty.head(top_n)


# =========================================================
# 20. Run debug
# =========================================================

print("\nAvailable debug output folder:")
print(OUTPUT_DIR)

user_input = input("\nEnter one item name for feature debug: ").strip()

debug_single_item(user_input, top_n=20)

print("\nSaved debug CSV files:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file)