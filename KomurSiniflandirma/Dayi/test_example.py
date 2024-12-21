import re
from playwright.sync_api import Page, expect

def test_has_title(page: Page):
    page.goto("http://localhost:8501/")

    # Expect a title "to contain" a substring.
    expect(page).to_have_title(re.compile("Kömür Sınıflandırma Uygulaması"))

def test_get_started_link(page: Page):
    page.goto("http://localhost:8501/")

    # Click the get started link.
    page.get_by_role("button", name="Browse files").click()

    # Expects page to have a heading with the name of Installation.
    expect(page.get_by_role("span", name="Kömür Sınıflandırma Uygulaması |")).to_be_visible()