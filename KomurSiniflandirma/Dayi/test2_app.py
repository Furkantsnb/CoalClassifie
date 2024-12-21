import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("http://localhost:8501/")
    expect(page.get_by_role("heading", name="Kömür Sınıflandırma Uygulamas")).to_be_visible()
    expect(page.get_by_text("Kömür Sınıflandırma Uygulamas")).to_be_visible()
    expect(page.get_by_text("Lütfen bir Kömür görüntüsü yü")).to_be_visible()
    expect(page.get_by_text("Kömür Görüntüsünü Yükleyin")).to_be_visible()
    expect(page.get_by_test_id("stFileUploaderDropzone")).to_be_visible()
    expect(page.get_by_test_id("baseButton-secondary")).to_be_visible()
    page.get_by_test_id("baseButton-secondary").click()
    page.get_by_test_id("baseButton-secondary").set_input_files("https://cdn.britannica.com/51/127751-004-97D5367E/Anthracite.jpg")
    # expect(page.get_by_test_id("stFileUploaderFile")).to_be_visible()
    # expect(page.get_by_test_id("baseButton-minimal")).to_be_visible()
    # expect(page.get_by_role("img", name="0")).to_be_visible()
    # expect(page.get_by_text("Tahmin Edilen Sınıf: Peat")).to_be_visible()
    # expect(page.get_by_text("Tahmin İhtimalleri:")).to_be_visible()
    # expect(page.get_by_text("AnthraciteBituminousLignitePeat020406080Tahmin İhtimalleri (Çubuk Grafik)SınıfY")).to_be_visible()
    # expect(page.get_by_text("78.2%14.6%5.43%1.71%PeatBituminousAnthraciteLigniteTahmin İhtimalleri (Pasta")).to_be_visible()
    # page.get_by_role("img", name="0").click()
    # page.get_by_test_id("StyledFullScreenButton").first.click()
    # page.get_by_role("button", name="Exit fullscreen").click()
    # page.get_by_test_id("StyledFullScreenButton").nth(1).click()
    # page.get_by_role("button", name="Exit fullscreen").click()
    # page.get_by_test_id("StyledFullScreenButton").nth(2).click()
    # page.get_by_role("button", name="Exit fullscreen").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
