import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger

# Import config and recommendation function
from config import MAIN_DATA_PATH
from step_05_basket_recommendation import recommend_for_cart

app = Flask(__name__)
# Configure Swagger UI
app.config['SWAGGER'] = {
    'title': 'Supershop Recommendation API',
    'uiversion': 3
}
swagger = Swagger(app)

# ==========================================
# 1. Load Data Mappings (ID <-> Name <-> Category)
# ==========================================
id_to_name = {}
name_to_id = {}
name_to_category = {}


def load_mappings():
    global id_to_name, name_to_id, name_to_category
    try:
        # Load main dataset to map original itemid and categories
        df = pd.read_csv(MAIN_DATA_PATH)

        # Helper to flexibly find column names
        def find_col(possible_names):
            lower_map = {c.lower().strip(): c for c in df.columns}
            for name in possible_names:
                key = name.lower().strip()
                if key in lower_map: return lower_map[key]
            return None

        item_id_col = find_col(["itemid", "item_id", "itemId"])
        item_name_col = find_col(["item_name", "itemName", "product_name"])
        cat_col = find_col(["category", "categoryName", "family"])

        if item_id_col and item_name_col:
            # Build dictionaries
            for _, row in df.iterrows():
                iid = str(row[item_id_col]).strip()
                iname = str(row[item_name_col]).strip()
                cat = str(row[cat_col]).strip() if cat_col else "General"

                if iname and iname != "nan":
                    id_to_name[iid] = iname
                    name_to_id[iname] = int(float(iid)) if iid.replace('.', '', 1).isdigit() else iid
                    name_to_category[iname] = cat
            print(f"Loaded {len(id_to_name)} item mappings successfully.")
        else:
            print("Warning: itemId or itemName column not found in main data.")

    except Exception as e:
        print(f"Error loading mappings: {e}")


# Call mapping loader on startup
load_mappings()


# ==========================================
# 2. API Endpoint Definition
# ==========================================
@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Get Basket Recommendations based on Customer Cart
    ---
    tags:
      - Recommendations
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            customerid:
              type: integer
              example: 23445
            date and time:
              type: string
              example: "2026-04-29 17:03:00"
            items:
              type: array
              items:
                type: object
                properties:
                  itemid:
                    type: integer
                    example: 952
                  quantity:
                    type: integer
                    example: 1
    responses:
      200:
        description: A list of recommended products and their details
    """
    req_data = request.get_json()

    if not req_data or "items" not in req_data:
        return jsonify({"error": "Invalid request. 'items' array is required."}), 400

    # Extract Item IDs from request
    cart_items = req_data.get("items", [])
    item_ids = [str(item.get("itemid")).strip() for item in cart_items if item.get("itemid")]

    # Map IDs to Item Names
    input_item_names = []
    for iid in item_ids:
        if iid in id_to_name:
            input_item_names.append(id_to_name[iid])
        # If float format string match fails (e.g. 952 vs 952.0)
        elif f"{iid}.0" in id_to_name:
            input_item_names.append(id_to_name[f"{iid}.0"])

    # If no items matched
    if not input_item_names:
        return jsonify({
            "input_item_names": [],
            "recommendations": [],
            "message": "None of the input item IDs matched the catalog."
        }), 200

    # Call your Recommendation Engine
    try:
        rec_df = recommend_for_cart(input_item_names, top_k=10)
    except Exception as e:
        return jsonify({"error": f"Error generating recommendations: {str(e)}"}), 500

    # Format the Output
    recommendations = []
    if not rec_df.empty:
        for _, row in rec_df.iterrows():
            rec_name = row["recommended_item_name"]
            rec_cat = name_to_category.get(rec_name, row.get("recommended_family", "General"))
            rec_id = name_to_id.get(rec_name, 0)

            recommendations.append({
                "category": rec_cat,
                "item_name": rec_name,
                "itemid": rec_id,
                "score": round(row["final_score"], 6)
            })

    response = {
        "input_item_names": input_item_names,
        "recommendations": recommendations
    }

    return jsonify(response)


if __name__ == '__main__':
    print("\n🚀 Starting Recommendation API Server...")
    print("👉 Swagger UI Documentation: http://127.0.0.1:5000/apidocs/")
    app.run(host='0.0.0.0', port=5000, debug=True)