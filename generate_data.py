import json
import os
from datetime import datetime, timedelta

def create_message(msg_id, account_id, date, channel, text):
    return {
        "message_id": msg_id,
        "account_id": account_id,
        "date": date.isoformat() + "Z",
        "channel": channel,
        "text": text
    }

def generate_innovate_messages(start_date):
    """Stable and healthy account. Routine interactions, consistently positive."""
    account_id = "acct_innovate"
    messages = []
    
    texts = [
        ("email", "Hi team, great kickoff meeting yesterday. Looking forward to getting started."),
        ("support_ticket", "Can you help me reset my password? Thanks!"),
        ("email", "Password reset worked. The new dashboard looks fantastic, by the way."),
        ("meeting_notes", "Monthly sync: Client is happy with progress. No blockers. Requested a minor color tweak to the reports."),
        ("email", "Thanks for turning around that color tweak so quickly!"),
        ("support_ticket", "We're seeing a slight delay in data sync this morning. Can you check?"),
        ("support_ticket", "Disregard last message, looks like it was on our end. All good!"),
        ("email", "The Q1 review report was spot on. Our VP loved it."),
        ("meeting_notes", "Monthly sync: Everything running smoothly. Discussed potential expansion next quarter."),
        ("email", "Can we schedule a call next week to discuss adding 5 more seats?"),
        ("email", "Great call today. Let's proceed with the license expansion."),
        ("support_ticket", "Need help assigning the new seats to the marketing team."),
        ("email", "All seats assigned. The onboarding flow was super smooth. Thanks!"),
        ("meeting_notes", "Quarterly Business Review: Client rated us 10/10. Renewing contract early."),
        ("email", "Signed the renewal contract. Thanks for a great first half of the year!")
    ]
    
    for i, (channel, text) in enumerate(texts):
        date = start_date + timedelta(days=(i * 11) + (i % 3))
        messages.append(create_message(f"msg_{account_id}_{i:03d}", account_id, date, channel, text))
        
    return messages

def generate_techcorp_messages(start_date):
    """Declining account. Starts good, but scope disputes and delays cause frustration."""
    account_id = "acct_techcorp"
    messages = []
    
    texts = [
        ("email", "Excited to partner with you on this integration. The proposed timeline looks good."),
        ("meeting_notes", "Kickoff: Alignment on phase 1 deliverables. Client seemed optimistic."),
        ("email", "Just checking in, are we still on track for the Friday delivery?"),
        ("email", "Understood that there is a slight delay. Please make sure we hit Wednesday next week."),
        ("support_ticket", "The Phase 1 build is missing the custom API endpoints we discussed."),
        ("email", "Your team is saying the custom API is out of scope. We explicitly discussed this during the sales process. This is unacceptable."),
        ("meeting_notes", "Escalation call: Client is frustrated about scope. Agreed to build 2 of the 5 endpoints at no cost. Client is not fully satisfied but accepted."),
        ("email", "Thanks for the compromise, but we are still behind schedule. When will Phase 2 start?"),
        ("support_ticket", "System threw a 500 error when we tried to test the new endpoints. Please fix ASAP."),
        ("email", "It has been 48 hours and the endpoints are still down. We have our own deadlines to meet."),
        ("email", "Another week, another delay. I need to escalate this to your leadership. Who is the VP of Customer Success?"),
        ("meeting_notes", "Executive sync: Client threatened to cancel. We promised a dedicated engineer to fix the issues."),
        ("support_ticket", "The dedicated engineer hasn't responded to my last three emails."),
        ("email", "This implementation is a disaster. We are spending more time managing your team than doing our jobs. We are reviewing our contract termination clauses."),
        ("email", "We are officially pausing all payments until the agreed upon scope is delivered in full.")
    ]
    
    for i, (channel, text) in enumerate(texts):
        days_to_add = 15 - i 
        if i == 0:
            date = start_date
        else:
            date = date + timedelta(days=max(2, days_to_add))
        messages.append(create_message(f"msg_{account_id}_{i:03d}", account_id, date, channel, text))
        
    return messages

def generate_cloudsync_messages(start_date):
    """Bumpy but recovering account. Has a major issue in the middle but resolves it."""
    account_id = "acct_cloudsync"
    messages = []
    
    texts = [
        ("email", "Welcome to the platform! We are ready to start."),
        ("meeting_notes", "Initial sync went well. Client is slowly migrating users."),
        ("email", "Migration of the first 100 users was successful. Thanks for the documentation."),
        ("support_ticket", "We are getting reports of slow load times from the migrated users."),
        ("support_ticket", "URGENT: The entire system is locked up. Nobody can access their files."),
        ("email", "What is the status of the outage? Our CEO is asking."),
        ("email", "This is causing massive disruption. We need a post-mortem on why this happened."),
        ("meeting_notes", "Post-mortem meeting: Explained the database indexing failure. Client appreciated the transparency but demands a credit for the downtime."),
        ("email", "Received the SLA credit. We appreciate you making this right quickly."),
        ("support_ticket", "Just a minor bug: the export button is greyed out for some admins."),
        ("email", "Thanks for the quick fix on the export bug."),
        ("meeting_notes", "Monthly sync: Stability has been great since the incident. Trust is rebuilding."),
        ("email", "We are ready to resume the migration for the remaining 500 users."),
        ("support_ticket", "Migration complete. No issues this time!"),
        ("email", "Really appreciate the white-glove support during this second migration phase. Great job team.")
    ]
    
    for i, (channel, text) in enumerate(texts):
        date = start_date + timedelta(days=(i * 10))
        if 3 <= i <= 6:
            date = messages[-1]["date"]
            prev_date = datetime.fromisoformat(date.replace("Z", ""))
            date = prev_date + timedelta(hours=12)
            
        messages.append(create_message(f"msg_{account_id}_{i:03d}", account_id, date, channel, text))
        
    return messages

def generate_dataflow_messages(start_date):
    """Sudden churn risk. Very quiet and normal, then suddenly severe issues."""
    account_id = "acct_dataflow"
    messages = []
    
    texts = [
        ("email", "Contract signed. Let's get going."),
        ("meeting_notes", "Quick kickoff. Client doesn't need much handholding. They will self-serve."),
        ("support_ticket", "How do I invite users?"),
        ("email", "Thanks, found it in the docs."),
        ("meeting_notes", "Quarterly check-in: Client says everything is fine. Very brief call, lasting 5 minutes."),
        ("email", "Please update our billing contact to Sarah."),
        ("support_ticket", "API rate limit increased?"),
        ("email", "Thanks for the limit increase."),
        ("meeting_notes", "Check-in: Still doing fine. No new requests."),
        ("email", "Can you send the latest compliance report?"),
        ("email", "Thanks for the report."),
        ("support_ticket", "CRITICAL: Security vulnerability detected in your agent integration. Our InfoSec team is blocking your application."),
        ("email", "We cannot use your software until a full independent audit is provided. This is a massive breach of our vendor requirements."),
        ("meeting_notes", "Emergency call: InfoSec completely blocked usage. They are demanding a remediation plan within 24 hours or they will rip us out."),
        ("email", "Your proposed remediation plan is insufficient. We are migrating to a competitor immediately.")
    ]
    
    for i, (channel, text) in enumerate(texts):
        date = start_date + timedelta(days=(i * 12))
        if i >= 11:
            prev_date = datetime.fromisoformat(messages[-1]["date"].replace("Z", ""))
            date = prev_date + timedelta(hours=24)
            
        messages.append(create_message(f"msg_{account_id}_{i:03d}", account_id, date, channel, text))
        
    return messages

def main():
    start_date = datetime(2023, 1, 1, 9, 0, 0)
    
    all_messages = []
    all_messages.extend(generate_innovate_messages(start_date))
    all_messages.extend(generate_techcorp_messages(start_date))
    all_messages.extend(generate_cloudsync_messages(start_date))
    all_messages.extend(generate_dataflow_messages(start_date))
    
    all_messages.sort(key=lambda x: x["date"])
    
    os.makedirs("data", exist_ok=True)
    
    with open("data/client_messages.json", "w") as f:
        json.dump(all_messages, f, indent=2)
        
    print(f"Generated {len(all_messages)} messages across 4 accounts.")
    print("Saved to data/client_messages.json")

if __name__ == "__main__":
    main()
