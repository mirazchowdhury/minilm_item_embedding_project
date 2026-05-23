# MiniLM Item Embedding Project

## Project Overview

MiniLM Item Embedding Project is a Python based retail recommendation system that creates product item embeddings from basket purchase data and uses those embeddings to recommend related items. The project reads customer basket data, builds a unique item catalog, generates text embeddings with the Hugging Face Sentence Transformers MiniLM model, builds basket and item relationship matrices, and serves recommendation results through scripts or a Flask API.

The repository is useful for grocery, retail, and supershop recommendation experiments where the input is a list of items already present in a customer basket and the output is a ranked list of relevant products.

## Main Objective

The main objective of this project is to recommend products that are related to a customer's current shopping basket. The system combines item text similarity, basket co occurrence, item user behavior, SVD based embeddings, rule based filtering, and intent based scoring to produce useful product recommendations.

## Repository Link

```text
https://github.com/mirazchowdhury/minilm_item_embedding_project
```

## Key Features

1. Builds a unique product catalog from basket purchase history.
2. Generates product text embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
3. Saves item embedding metadata for later inspection.
4. Builds sparse basket one hot and co occurrence matrices.
5. Creates item user matrices from customer purchase records.
6. Applies Truncated SVD to generate compact item representations.
7. Uses cosine similarity for item relation scoring.
8. Adds business rule based filtering for grocery recommendation quality.
9. Detects basket intent from multiple input products.
10. Reduces wrong same family suggestions where they are not useful.
11. Supports command line testing for recommendation output.
12. Provides Flask API support with Swagger UI.
13. Separates training pipeline files, API files, data files, and model artifacts.

## Technology Stack

```text
Python
Pandas
NumPy
SciPy
Scikit learn
Sentence Transformers
PyTorch
Hugging Face models
Truncated SVD
Cosine similarity
Flask
Flasgger Swagger UI
Batch scripts for Windows execution
```

## Main Folder Structure

```text
minilm_item_embedding_project
    data
        input
            customer_purchase_with_basket_list.csv
            form_paste.csv
            form_paste_clean_v2 .csv
            form_paste_clean_v2_baskets_list .csv
    models
    src
        app.py
        basket_intent_engine.py
        basket_recommender.py
        business_rule_seed.py
        config.py
        cuda_check.py
        item_intelligence.py
        run_pipeline.py
        step_01_create_item_catalog.py
        step_02_generate_minilm_embeddings.py
        step_03_build_topk_recommendations.py
        step_04_test_recommendation.py
        step_05_basket_recommendation.py
        step_06_feature_engineering_debug_outputs.py
        step_07_xgboost_ranker.py
        utils.py
    minilm_item_embedding_project_api
        app.py
        requirements.txt
        data
        models
        src
            api_recommendation_rules.py
            api_recommender_service.py
    INTENT_RECOMMENDER_NOTES.txt
    LICENSE
    README.md
    requirements.txt
    run_pipeline.bat
    test_recommendation.bat
```

## Dataset Files

The project expects input files inside the `data/input` directory.

### Main basket file

```text
data/input/customer_purchase_with_basket_list.csv
```

This file is used to build the item catalog and basket co occurrence features. The expected basket column name is configured as:

```text
items
```

### Main transaction file

```text
data/input/form_paste_clean_v2 .csv
```

This file is used for item user matrix creation and API mapping. The configured columns include:

```text
customerId
itemName
```

The file name contains a space before `.csv`. Keep the same name or update `src/config.py` before running the project.

## Configuration

The main configuration file is:

```text
src/config.py
```

Main configuration values include:

```text
PROJECT_ROOT
DATA_INPUT_DIR
OUTPUT_DIR
MODEL_DIR
BASKET_DATA_PATH
MAIN_DATA_PATH
ITEM_CATALOG_PATH
TEXT_EMBEDDING_MATRIX_PATH
TEXT_EMBEDDING_METADATA_PATH
BASKET_ONE_HOT_MATRIX_PATH
ITEM_USER_MATRIX_PATH
COOCCURRENCE_MATRIX_PATH
ITEM_USER_SVD_EMBEDDING_PATH
COOCCURRENCE_SVD_EMBEDDING_PATH
FEATURE_ITEM_INDEX_PATH
ITEM_POPULARITY_PATH
TOPK_RECOMMENDATION_PATH
MODEL_NAME
BATCH_SIZE
NORMALIZE_EMBEDDINGS
ITEMS_COLUMN
CUSTOMER_COLUMN
ITEM_NAME_COLUMN
SVD_DIM
TOP_K
CANDIDATE_POOL_SIZE
```

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Default embedding batch size:

```text
128
```

Default output recommendation count:

```text
5
```

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/mirazchowdhury/minilm_item_embedding_project.git
cd minilm_item_embedding_project
```

### Step 2: Create a virtual environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install pandas numpy scipy scikit-learn sentence-transformers torch tqdm joblib
```

The repository requirements file includes these main packages:

```text
pandas
numpy
scikit-learn
sentence-transformers
torch
tqdm
joblib
scipy
huggingface-hub
```

### Step 4: Optional CUDA setup

For a CUDA supported NVIDIA GPU, install the PyTorch CUDA build that matches your system.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check CUDA availability:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## How the Pipeline Works

### Step 1: Create item catalog

Script:

```text
src/step_01_create_item_catalog.py
```

Purpose:

1. Reads basket data from `customer_purchase_with_basket_list.csv`.
2. Parses product names from the `items` column.
3. Counts product frequency across baskets.
4. Builds a unique item catalog.
5. Creates clean product text for embedding generation.
6. Saves the catalog to `data/output/item_catalog_from_basket.csv`.

Output:

```text
data/output/item_catalog_from_basket.csv
```

### Step 2: Generate MiniLM embeddings

Script:

```text
src/step_02_generate_minilm_embeddings.py
```

Purpose:

1. Loads the item catalog.
2. Reads the `product_text` column.
3. Loads the MiniLM sentence transformer model.
4. Generates normalized product text embeddings.
5. Saves the embedding matrix and metadata.

Outputs:

```text
models/product_text_embeddings_minilm.npy
data/output/product_text_embedding_metadata.csv
```

### Step 3: Build top K recommendations

Script:

```text
src/step_03_build_topk_recommendations.py
```

Purpose:

1. Loads the item catalog.
2. Loads basket records.
3. Builds a basket one hot matrix.
4. Builds item co occurrence matrix.
5. Builds item user matrix from customer transactions.
6. Applies Truncated SVD to item user and co occurrence matrices.
7. Loads MiniLM text embeddings.
8. Combines model scores and rule based scores.
9. Saves final ranked recommendations.

Main techniques used:

1. Sparse matrix construction.
2. Item pair counting.
3. Co occurrence score.
4. Item user behavior score.
5. SVD embedding score.
6. Text embedding similarity score.
7. Intent compatibility score.
8. Product family rules.
9. Same type product penalty.
10. Hard blocking for unrelated products.

Expected outputs:

```text
models/basket_one_hot_matrix.npz
models/cooccurrence_matrix.npz
models/item_user_matrix.npz
models/item_user_svd_embeddings.npy
models/cooccurrence_svd_embeddings.npy
data/output/feature_item_index.csv
data/output/item_popularity.csv
data/output/topk_hybrid_recommendations.csv
```

### Step 4: Test recommendation

Script:

```text
src/step_04_test_recommendation.py
```

Purpose:

1. Loads generated recommendation output.
2. Tests related item lookup.
3. Prints recommendations for selected products.

### Step 5: Basket level recommendation

Script:

```text
src/step_05_basket_recommendation.py
```

Purpose:

1. Accepts multiple products as cart input.
2. Detects basket intent.
3. Applies token based matching and fuzzy matching.
4. Blocks unrelated cleaning, personal care, toy, and other products when the basket is food focused.
5. Reduces repeated same family recommendations.
6. Returns ranked basket recommendations.

Example input cases from project notes:

```text
Beef Bone Full
Beef Bone Full, Onion Deshi(New)-kg
Pusti Suji -500g, Akher Sugar -1kg
rice, potato
```

## Run the Project

### Run the complete pipeline

```bash
python src/run_pipeline.py
```

For Windows:

```bash
run_pipeline.bat
```

### Run each step manually

```bash
python src/step_01_create_item_catalog.py
python src/step_02_generate_minilm_embeddings.py
python src/step_03_build_topk_recommendations.py
python src/step_04_test_recommendation.py
```

### Run basket recommendation test

```bash
python src/step_05_basket_recommendation.py
```

For Windows:

```bash
test_recommendation.bat
```

## Output Files

Main generated files:

```text
data/output/item_catalog_from_basket.csv
data/output/product_text_embedding_metadata.csv
data/output/feature_item_index.csv
data/output/item_popularity.csv
data/output/topk_hybrid_recommendations.csv
models/product_text_embeddings_minilm.npy
models/basket_one_hot_matrix.npz
models/cooccurrence_matrix.npz
models/item_user_matrix.npz
models/item_user_svd_embeddings.npy
models/cooccurrence_svd_embeddings.npy
```

API specific generated or required files:

```text
minilm_item_embedding_project_api/data/output/item_catalog.csv
minilm_item_embedding_project_api/models/item_index.csv
minilm_item_embedding_project_api/models/debug_co_occurrence_matrix.npz
minilm_item_embedding_project_api/models/debug_main_item_user_svd_embedding.npy
minilm_item_embedding_project_api/models/debug_backup_co_occurrence_svd_embedding.npy
minilm_item_embedding_project_api/models/debug_minilm_text_embedding.npy
```

## Flask API From Main Source Folder

The main source folder includes a Flask app:

```text
src/app.py
```

It exposes this endpoint:

```text
POST /api/recommend
```

Swagger UI:

```text
http://127.0.0.1:5000/apidocs/
```

Run command:

```bash
python src/app.py
```

Example request:

```json
{
  "customerid": 23445,
  "date and time": "2026-04-29 17:03:00",
  "items": [
    {
      "itemid": 952,
      "quantity": 1
    }
  ]
}
```

Example response format:

```json
{
  "input_item_names": [
    "Example Product Name"
  ],
  "recommendations": [
    {
      "category": "Example Category",
      "item_name": "Recommended Product",
      "itemid": 123,
      "score": 0.876543
    }
  ]
}
```

## Separate API Folder

The repository also contains a separate API project:

```text
minilm_item_embedding_project_api
```

API application:

```text
minilm_item_embedding_project_api/app.py
```

Available endpoints:

```text
GET /health
POST /recommend
```

Run command:

```bash
cd minilm_item_embedding_project_api
pip install -r requirements.txt
python app.py
```

Health check:

```text
http://127.0.0.1:5000/health
```

Swagger UI:

```text
http://127.0.0.1:5000/apidocs/
```

Example request for `/recommend`:

```json
{
  "customerid": 23445,
  "date and time": "2026-04-29 17:03:00",
  "top_n": 10,
  "items": [
    {
      "item_id": 952,
      "quantity": 1
    }
  ]
}
```

Expected response format:

```json
{
  "input_item_names": [
    "Example Product Name"
  ],
  "missing_item_ids": [],
  "detected_cart_intent": "daily_cooking",
  "recommendations": [
    {
      "category": "Example Category",
      "item_name": "Recommended Product",
      "item_id": 123,
      "score": 0.876543
    }
  ]
}
```

## API Implementation Note

In the API folder, the app imports:

```python
from src.api_recommender_service import RecommenderService
```

The service file shows a class named:

```python
BasketRecommenderService
```

If the API raises an import error, update the import and service initialization to use the available class name, or create an alias inside `api_recommender_service.py`.

Example alias:

```python
RecommenderService = BasketRecommenderService
```

Also check the item key name. The API documentation shows `itemid`, while the service reads `item_id`. Use `item_id` for the separate API folder unless the service code is changed.

## Recommendation Logic

The recommendation score is built from several signals.

1. Item user similarity from customer purchase history.
2. Backup co occurrence embedding similarity.
3. MiniLM text embedding similarity.
4. Pair count score from basket co occurrence.
5. Rule based boost from detected basket intent.
6. Same type item penalty.
7. Hard block rule for unrelated products.

In the API service, the model score combines these weighted signals:

```text
0.30 item user similarity
0.25 co occurrence embedding similarity
0.15 MiniLM text similarity
0.30 pair count score
```

Then rule based boost and penalties are applied before ranking the final recommendations.

## Why MiniLM Is Used

MiniLM is used to represent product text as dense vectors. This helps the system understand similarity between item names even when exact co purchase history is weak. For example, different rice products or different milk products may be close in embedding space because their text descriptions are semantically similar.

However, text similarity alone is not enough for basket recommendation. A product like rice should not only recommend another rice item. A better system should also consider complementary items such as dal, oil, onion, egg, meat, fish, spice, and vegetables. This repository addresses that issue by combining text embeddings with basket co occurrence and rule based intent logic.

## Business Rule Examples

The project includes rule based logic for several product groups and shopping intents.

Example groups:

```text
raw meat
raw fish
vegetable
fruit
spice
staple
pulse
oil
dairy
bakery
snack
beverage
sauce
cleaning
personal care
household
baby
toy stationery
clothing beauty
```

Example use cases:

1. If the basket contains beef or fish, the system can prefer spice, oil, onion, garlic, ginger, rice, and sauce type items.
2. If the basket contains dessert items such as suji or semai, the system can prefer sugar, milk, ghee, raisin, cashew, and related dessert items.
3. If the basket is food focused, the system avoids unrelated detergent, toy, cleaning, and personal care items.
4. If a candidate is too similar to the already selected item, the system can penalize it to avoid repeated same type results.

## Common Issues and Fixes

### Item catalog not found

Error example:

```text
Item catalog not found
```

Fix:

```bash
python src/step_01_create_item_catalog.py
```

### MiniLM embedding file not found

Error example:

```text
MiniLM embedding file not found
```

Fix:

```bash
python src/step_02_generate_minilm_embeddings.py
```

### Main dataset not found

Check this path in `src/config.py`:

```text
data/input/form_paste_clean_v2 .csv
```

The file name includes a space before `.csv`.

### API asset files not found

The separate API folder expects files such as:

```text
item_catalog.csv
item_index.csv
debug_co_occurrence_matrix.npz
debug_main_item_user_svd_embedding.npy
debug_backup_co_occurrence_svd_embedding.npy
debug_minilm_text_embedding.npy
```

Generate or copy the needed outputs into the API folder before running the API.

### CUDA not detected

Run:

```python
import torch
print(torch.cuda.is_available())
```

If it prints `False`, install the correct PyTorch version for your CUDA setup or run the model on CPU.

## Suggested Development Workflow

1. Put the transaction and basket CSV files inside `data/input`.
2. Check column names inside `src/config.py`.
3. Run catalog generation.
4. Run MiniLM embedding generation.
5. Run recommendation building.
6. Test recommendation output using the command line script.
7. Copy final model artifacts into the API folder if needed.
8. Run the Flask API.
9. Test API output from Swagger UI or Postman.

## Example End To End Commands

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scipy scikit-learn sentence-transformers torch tqdm joblib
python src/step_01_create_item_catalog.py
python src/step_02_generate_minilm_embeddings.py
python src/step_03_build_topk_recommendations.py
python src/step_05_basket_recommendation.py
python src/app.py
```

## Project Strengths

1. Combines semantic product similarity with real basket behavior.
2. Uses sparse matrices for large item basket data.
3. Adds business rules to avoid irrelevant recommendations.
4. Supports both script based use and API based use.
5. Keeps generated artifacts in data and model directories.
6. Provides a practical starting point for retail recommendation systems.

## Future Improvements

1. Add a single clean training pipeline that also prepares the API assets.
2. Rename files with spaces to reduce path issues.
3. Standardize API request keys to use either `itemid` or `item_id` everywhere.
4. Add unit tests for item parsing, intent detection, and API response format.
5. Add Docker support for API deployment.
6. Add evaluation metrics such as hit rate, recall at K, precision at K, and mean reciprocal rank.
7. Add popularity fallback for cold start products.
8. Add product category hierarchy support.
9. Add database support for real time retail use.
10. Add automated model artifact versioning.

## License

This repository is released under the MIT License.

## Author

Repository owner:

```text
mirazchowdhury
```

GitHub repository:

```text
https://github.com/mirazchowdhury/minilm_item_embedding_project
```
