import unittest

from fastapi.testclient import TestClient

from app.main import app


class MainRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_returns_service_status(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"service": "Tripletex Agent", "status": "ok"})

    def test_favicon_returns_no_content(self):
        response = self.client.get("/favicon.ico")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")


if __name__ == "__main__":
    unittest.main()
