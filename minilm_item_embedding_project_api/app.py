from pathlib import Path
from flask import Flask, request, jsonify
from flasgger import Swagger
from src.api_recommender_service import RecommenderService

PROJECT_ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Retail Basket Recommendation API",
        "description": "Intent-based basket recommendation API",
        "version": "1.0.0"
    }
}

swagger = Swagger(app, template=swagger_template)
recommender_service = RecommenderService(PROJECT_ROOT)

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: API health status
    """
    return jsonify({
        "status": "ok",
        "message": "Recommendation API is running"
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Basket recommendation endpoint
    ---
    tags:
      - Recommendation
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - customerid
            - date and time
            - items
          properties:
            customerid:
              type: integer
              example: 23445
            date and time:
              type: string
              example: "2026-04-29 17:03:00"
            top_n:
              type: integer
              example: 10
            items:
              type: array
              items:
                type: object
                required:
                  - itemid
                  - quantity
                properties:
                  itemid:
                    type: integer
                    example: 952
                  quantity:
                    type: integer
                    example: 1
    responses:
      200:
        description: Recommendation result
      400:
        description: Invalid request
      500:
        description: Internal server error
    """
    try:
        payload = request.get_json(force=True)

        if payload is None:
            return jsonify({
                "error": "Invalid JSON body"
            }), 400

        if "items" not in payload:
            return jsonify({
                "error": "items field is required"
            }), 400

        if not isinstance(payload["items"], list):
            return jsonify({
                "error": "items must be a list"
            }), 400

        top_n = int(payload.get("top_n", 10))

        result = recommender_service.recommend(
            payload, top_n=top_n
        )

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )