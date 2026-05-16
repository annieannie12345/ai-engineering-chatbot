import unittest

from app.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_load_without_env_file(self):
        settings = Settings.from_env()

        self.assertTrue(settings.ollama_base_url.startswith("http"))
        self.assertGreater(settings.retriever_top_k, 0)
        self.assertGreater(settings.chunk_size, settings.chunk_overlap)


if __name__ == "__main__":
    unittest.main()
