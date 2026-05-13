# MiniLM Item Embedding Project

এই project টি PyCharm এ open করে সরাসরি চালানো যাবে। এটি basket dataset থেকে unique item catalog বানাবে, Hugging Face এর `sentence-transformers/all-MiniLM-L6-v2` model দিয়ে product text embedding তৈরি করবে, তারপর cosine similarity দিয়ে top related item save করবে।

## Folder structure

```text
minilm_item_embedding_project
    data
        input
            customer_purchase_with_basket_list.csv
        output
    models
    logs
    src
        config.py
        utils.py
        step_01_create_item_catalog.py
        step_02_generate_minilm_embeddings.py
        step_03_build_topk_recommendations.py
        step_04_test_recommendation.py
        run_pipeline.py
    requirements.txt
    run_pipeline.bat
    test_recommendation.bat
```

## PyCharm setup

1. PyCharm open করুন।
2. Open project থেকে এই folder select করুন।
3. Terminal open করে virtual environment তৈরি করুন।

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## CUDA setup

যদি আপনার CUDA supported Nvidia GPU থাকে, তাহলে PyTorch CUDA version install করুন। Example command নিচে দেওয়া হলো। আপনার CUDA version অনুযায়ী official PyTorch website থেকে command মিলিয়ে নিতে পারেন।

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA check করার জন্য:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Run full pipeline

PyCharm terminal থেকে:

```bash
python src\run_pipeline.py
```

অথবা Windows এ:

```bash
run_pipeline.bat
```

## Run step by step

```bash
python src\step_01_create_item_catalog.py
python src\step_02_generate_minilm_embeddings.py
python src\step_03_build_topk_recommendations.py
python src\step_04_test_recommendation.py
```

## Output files

```text
data/output/item_catalog_from_basket.csv
data/output/product_text_embedding_metadata.csv
data/output/topk_text_embedding_recommendations.csv
models/product_text_embeddings_minilm.npy
```

## Important note

এই embedding product name similarity ধরবে। Rice দিলে অনেক সময় অন্য rice type বা similar grocery product আসতে পারে। Complementary recommendation, যেমন rice এর সাথে daal, chicken, onion, egg, masala, ভালো করতে হলে basket co purchase score, association rule, category rule, এবং popularity score add করা উচিত।
