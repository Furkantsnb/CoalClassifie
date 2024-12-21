from locust import HttpUser , task

class TestAppLocust(HttpUser):
    @task
    def test_app(self):
        self.client.get("/")