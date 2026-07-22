import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face.register_person import PersonRegister

registration = PersonRegister()

registration.register("Akshay")
