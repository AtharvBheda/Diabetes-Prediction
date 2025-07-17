from config import config

def validate_form_data(form_data):
    """Simple form validation"""
    errors = []
    
    for field, limits in config.FEATURE_LIMITS.items():
        if field not in form_data or not form_data[field]:
            errors.append(f"{field.replace('_', ' ').title()} is required")
            continue
            
        try:
            value = float(form_data[field])
            if value < limits['min'] or value > limits['max']:
                errors.append(f"{field.replace('_', ' ').title()} must be between {limits['min']} and {limits['max']}")
        except ValueError:
            errors.append(f"{field.replace('_', ' ').title()} must be a valid number")
    
    return len(errors) == 0, errors
