# Parking Violation System

from datetime import datetime

class ParkingViolation:
    def __init__(self, license_plate, violation_time, violation_location):
        self.license_plate = license_plate
        self.violation_time = violation_time
        self.violation_location = violation_location

    def display_violation(self):
        print(f"License Plate: {self.license_plate}")
        print(f"Violation Time: {self.violation_time}")
        print(f"Violation Location: {self.violation_location}")

# Example usage:
if __name__ == '__main__':
    violation = ParkingViolation('ABC123', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 'Lot A')
    violation.display_violation()