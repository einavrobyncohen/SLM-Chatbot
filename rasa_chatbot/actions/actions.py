import pickle
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionPredictTicketApproval(Action):

    def name(self) -> str:
        return "action_predict_ticket_approval"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        # Extract ticket type and priority from user input (entities)
        ticket_type = tracker.get_slot('ticket_type')
        priority = tracker.get_slot('priority')

        # Convert categorical feature (ticket_type) to numerical using label encoding or a simple mapping
        ticket_type_mapping = {
            "software": 0,
            "network": 1,
            "hardware": 2
        }

        # Handle missing or unseen ticket_type
        if ticket_type not in ticket_type_mapping:
            dispatcher.utter_message(text=f"Unknown ticket type: {ticket_type}.")
            return []

        # Get the numerical representation of the ticket_type
        ticket_type_num = ticket_type_mapping[ticket_type]

        # Load the trained ML model
        with open('../ml_model/model/ticket_approval_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Define the full feature set (dummy values for features not provided)
        input_data = {
            'Ticket Type': [ticket_type_num],  # Use the encoded ticket type
            'Priority': [priority],  # This assumes priority is already a number
            'breach': [0],
            'bug': [0],
            'caused': [0],
            'connectivity': [0],
            'detected': [0],
            'error': [0],
            'failure': [0],
            'high_priority': [0],
            'issue': [0],
            'low_priority': [0],
            'medium_priority': [0],
            'network_issue': [0],
            'outage': [0],
            'resolved': [0],
            'software_issue': [0]
        }

        # Create DataFrame for the input
        input_df = pd.DataFrame(input_data)

        # Make prediction (0 = No, 1 = Yes)
        prediction = model.predict(input_df)[0]

        # Respond with the prediction
        if prediction == 1:
            message = f"The ticket with type {ticket_type} and priority {priority} is likely to be approved."
        else:
            message = f"The ticket with type {ticket_type} and priority {priority} is likely to be rejected."

        dispatcher.utter_message(text=message)

        return []
