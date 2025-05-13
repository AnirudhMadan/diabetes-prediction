import unittest
from auth import add_user, authenticate_user
from database import create_user_table
import os

class AuthTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the database before any tests are run"""
        cls.db_path = 'test_db.sqlite'
        # Initialize the database
        create_user_table()

    @classmethod
    def tearDownClass(cls):
        """Clean up the database after tests are done"""
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def test_register_user(self):
        """Test the user registration process"""
        username = 'testuser'
        password = 'testpass'
        
        # Test registration
        result = add_user(username, password)
        self.assertTrue(result, "User should be registered successfully.")

        # Test that the username already exists (this should fail)
        result = add_user(username, password)
        self.assertFalse(result, "User registration should fail if the username already exists.")

    def test_authenticate_user(self):
        """Test the user authentication process"""
        username = 'testlogin'
        password = 'loginpass'
        
        # Register the user
        add_user(username, password)

        # Test successful authentication
        result = authenticate_user(username, password)
        self.assertTrue(result, "Authentication should succeed for valid credentials.")

        # Test failed authentication (wrong password)
        result = authenticate_user(username, 'wrongpassword')
        self.assertFalse(result, "Authentication should fail for incorrect password.")

if __name__ == "__main__":
    unittest.main()
