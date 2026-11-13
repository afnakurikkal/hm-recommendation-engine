H&M Personalized Product Recommendation API

1. Project Overview

This module implements a REST API for the H&M Personalized Product Recommendation System using FastAPI. It generates personalized product recommendations based on a customer's previous purchase history.

2. Day 1 – FastAPI Setup

The FastAPI application was created to provide recommendation services through HTTP endpoints.

Objectives

Set up the FastAPI application.

Create API endpoints for recommendation requests.

Accept customer ID as input.

Support configurable Top-K recommendations.

Return recommendation results in JSON format.

3. API Endpoints

MethodEndpointDescriptionGET/Checks whether the API is running.GET/recommend/{customer_id}Generates personalized product recommendations. 

Example Request

GET /recommend/{customer_id}?k=5
Here:

customer_id identifies the customer.

k specifies the number of recommendations required.

Default value of k is 5.

Maximum allowed value of k is 100.

4. Technologies Used

Python, FastAPI, Uvicorn, Pandas, NumPy, FAISS, and Redis

5. Recommendation Workflow

Receive the customer ID through the API.

Retrieve the customer's purchase history.

Generate a customer preference vector using purchased item embeddings.

Search for similar products using the FAISS HNSW index.

Exclude products already purchased by the customer.

Retrieve product details from Redis.

Return the recommendations as a JSON response.

6. Expected API Response

The recommendation endpoint returns:

Customer ID

Requested number of recommendations

Total recommendations generated

Recommended article IDs

Product details

7. Testing

The API was tested using Python requests.
The recommendation endpoint was successfully tested with k=10 and returned 10 product recommendations for a valid customer ID.

8. Result

The FastAPI recommendation service was successfully implemented and tested. It provides personalized Top-K product recommendations using customer purchase history, item embeddings, FAISS similarity search and Redis product metadata.
