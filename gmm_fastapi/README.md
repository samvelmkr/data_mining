# GMM FastAPI Project

This project implements a Gaussian Mixture Model (GMM) using FastAPI. It provides an endpoint to predict cluster probabilities based on input points in a 2D space.

## Requirements

Make sure you have Python 3.x installed. You can install the required packages using pip:

```bash
pip3 install -r requirements.txt
```

## Running the Server

To run the FastAPI server, use the following command:

```bash
uvicorn gmm_fastapi:app --reload
```

- This will start the server at `http://127.0.0.1:8000`.

## Sending Requests

You can send requests to the server using the provided `get_point_proba.py` script. This script allows you to input `x` and `y` values and will return the cluster probabilities from the GMM.

To use the script, run:

```bash
python3 get_point_proba.py
```

You will be prompted to enter values for `x` and `y`. The script will then send these values to the FastAPI endpoint and display the response.

## Endpoint

The FastAPI server exposes the following endpoint:

- **POST** `/predict/`: Accepts a JSON payload with `x` and `y` values, returning the probabilities for the clusters.

### Example Request Payload

```json
{
    "x": 0.5,
    "y": 0.6
}
```
