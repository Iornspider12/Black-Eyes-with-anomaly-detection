
import time
from twilio.rest import Client
from datetime import datetime

# Twilio credentials (replace with your real credentials)
account_sid = 'x'
auth_token = 'x'
from_number = 'x'  # Your Twilio number
to_number = 'x'    # Your personal phone number

client = Client(account_sid, auth_token)
alert_log = {}
ALERT_INTERVAL = 60

def check_and_alert(name):
    current_time = datetime.now()
    if name == 'Unknown':
        last_alert_time = alert_log.get(name)
        if not last_alert_time or (current_time - last_alert_time).total_seconds() > ALERT_INTERVAL:
            try:
                message = client.messages.create(
                    body=f"🚨 Unknown person detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')}.",
                    from_=from_number,
                    to=to_number
                )
                print(f"Twilio Alert Sent: SID {message.sid}")
                alert_log[name] = current_time
            except Exception as e:
                print(f"Twilio Error: {e}")
        else:
            print("⏱ Alert skipped (sent recently).")