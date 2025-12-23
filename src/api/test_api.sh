echo "Testing Health Endpoint..."
curl -s http://localhost:5001/health | python3 -m json.tool

echo -e "\n\nTesting Prediction (Billing Inquiry)..."
curl -s -X POST -H "Content-Type: application/json" \
    -d '{"ticket_id": "123", "subject": "Charge on my card", "description": "I was charged twice for the subscription."}' \
    http://localhost:5001/predict | python3 -m json.tool

echo -e "\n\nTesting Prediction (Technical Issue)..."
curl -s -X POST -H "Content-Type: application/json" \
    -d '{"ticket_id": "124", "subject": "Connection Error", "description": "Cannot connect to the database server."}' \
    http://localhost:5001/predict | python3 -m json.tool
