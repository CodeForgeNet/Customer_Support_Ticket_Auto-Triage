import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

CATEGORIES = [
    "Bug Report",
    "Feature Request",
    "Technical Issue",
    "Billing Inquiry",
    "Account Management"
]

PRIORITIES = ["Low", "Medium", "High", "Critical"]

TEMPLATES = {
    "Bug Report": {
        "subjects": [
            "App crashes on launch", "Button not working", "Error 404 when clicking link",
            "Data not saving", "Typos in dashboard", "Search returns wrong results",
            "Login fails with correct password", "Images not loading", "Slow performance on checkout",
            "Mobile view broken"
        ],
        "descriptions": [
            "Every time I try to open the app, it crashes immediately. I'm on iOS 15.",
            "The submit button is unresponsive. I clicked it multiple times but nothing happens.",
            "I'm getting a 404 error when I try to access my profile settings.",
            "I updated my profile but the changes are not being saved.",
            "There are several spelling mistakes in the main dashboard view.",
            "Searching for 'invoice' returns results for 'inventory'. This is confusing.",
            "I checked my password three times, but I still can't log in.",
            "Product images are showing as broken links on the gallery page.",
            "The checkout process is extremely slow, taking over a minute to load.",
            "The menu is cut off on my Android phone screen."
        ]
    },
    "Feature Request": {
        "subjects": [
            "Add dark mode", "Export to PDF", "Integration with Slack",
            "Two-factor authentication", "Customizable dashboard", "Bulk edit feature",
            "Undo button", "Mobile app version", "Voice commands", "API access"
        ],
        "descriptions": [
            "It would be great to have a dark mode for better night time viewing.",
            "We need the ability to export our monthly reports to PDF format.",
            "Can you add an integration with Slack for notifications?",
            "Please add 2FA for better security on our accounts.",
            "I want to be able to rearrange the widgets on my dashboard.",
            "It's tedious to edit items one by one. A bulk edit feature is needed.",
            "I accidentally deleted a file. An undo button would be a lifesaver.",
            "Is there a native mobile app planned? The web view is okay but an app would be better.",
            "Adding voice commands for navigation would be a cool accessibility feature.",
            "We need API access to pull data into our own internal tools."
        ]
    },
    "Technical Issue": {
        "subjects": [
            "Cannot connect to server", "API timeout", "Database connection error",
            "SSL certificate expired", "Integration failing", "Data sync issues",
            "Firewall blocking traffic", "DNS resolution failed", "Websocket disconnects",
            "High latency"
        ],
        "descriptions": [
            "I'm getting a 'Cannot connect to server' message when using the desktop client.",
            "The API is timing out after 30 seconds. Is there a service degradation?",
            "We are seeing database connection errors in our logs.",
            "The browser is warning that the SSL certificate has expired.",
            "Our integration with Salesforce has stopped working suddenly.",
            "The data between the mobile app and web dashboard is not syncing.",
            "I think our firewall is blocking the traffic to your authentication servers.",
            "We are unable to resolve the hostname for the API endpoint.",
            " The websocket connection keeps dropping every few minutes.",
            "We are experiencing very high latency from the Singapore region."
        ]
    },
    "Billing Inquiry": {
        "subjects": [
            "Invoice explanation", "Refund request", "Update credit card",
            "Downgrade subscription", "Charged twice", "Pro-rated charges",
            "Payment failed", "VAT invoice needed", "Cancel subscription",
            "Upgrade pricing"
        ],
        "descriptions": [
            "I don't understand the extra charge on my latest invoice.",
            "I would like to request a refund for the unused months of my subscription.",
            "My credit card expired, how do I update the payment method?",
            "I want to downgrade to the free tier as I don't use the premium features.",
            "I noticed I was charged twice for this month's subscription.",
            "How are pro-rated charges calculated if I upgrade mid-month?",
            "My payment failed but I have sufficient funds. verified with bank.",
            "I need a VAT invoice for my company's tax records.",
            "I need to cancel my subscription immediately.",
            "What is the pricing difference if I upgrade to the Enterprise plan?"
        ]
    },
    "Account Management": {
        "subjects": [
            "Reset password", "Change email address", "Delete account",
            "Add user to team", "Permission issues", "Account locked",
            "Merge accounts", "Update profile picture", "GDPR request",
            "Transfer ownership"
        ],
        "descriptions": [
            "I forgot my password and the reset link is not arriving.",
            "I need to change the email address associated with my account.",
            "Please permanently delete my account and all associated data.",
            "I need to add a new team member to our organization account.",
            "I can't access the admin panel even though I should have permissions.",
            "My account has been locked due to too many login attempts.",
            "I have two accounts, can you merge them into one?",
            "I'm unable to upload a new profile picture.",
            "I would like to request a copy of all my data under GDPR.",
            "I need to transfer ownership of the organization to another user."
        ]
    }
}

def generate_ticket():
    category = random.choice(CATEGORIES)
    template = TEMPLATES[category]
    subject = random.choice(template["subjects"])
    description = random.choice(template["descriptions"])
    
    if random.random() > 0.8:
        description += f" Please help! Ticket ID ref: {random.randint(1000, 9999)}."
    if random.random() > 0.9:
        subject = f"URGENT: {subject}"
        
    priority = random.choice(PRIORITIES)
    if "URGENT" in subject or "Critical" in subject or category == "Technical Issue":
         priority = np.random.choice(PRIORITIES, p=[0.1, 0.2, 0.3, 0.4])
    else:
         priority = np.random.choice(PRIORITIES, p=[0.4, 0.3, 0.2, 0.1])

    return {
        "ticket_id": str(uuid.uuid4()),
        "subject": subject,
        "description": description,
        "category": category,
        "priority": priority,
        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
    }

def main():
    print("Generating synthetic dataset...")
    data = [generate_ticket() for _ in range(5000)]
    df = pd.DataFrame(data)
    
    output_path = os.path.join("data", "raw", "tickets.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} tickets to {output_path}")
    
    print("\nSample Data:")
    print(df.head())
    print("\nCategory Distribution:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    main()
