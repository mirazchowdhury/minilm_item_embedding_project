from step_01_create_item_catalog import create_item_catalog
from step_02_generate_minilm_embeddings import generate_embeddings
from step_03_build_topk_recommendations import build_recommendations

def main():
    create_item_catalog()
    generate_embeddings()
    build_recommendations()

if __name__ == "__main__":
    main()