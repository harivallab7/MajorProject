from requests_html import HTMLSession, PyQuery as pq
from datetime import datetime, timedelta
import json
import os
import time

session = HTMLSession()

def scrape_news(start_date, end_date, output_file):
    dt = timedelta(days=1)
    allData = {}
    max_retries = 5  # Maximum number of retries for fetching articles

    current_date = start_date
    while current_date <= end_date:
        retries = 0
        while retries < max_retries:
            url = f'https://www.dawn.com/latest-news/{str(current_date.date())}'
            print(f"Fetching data from {url}")
            try:
                r = session.get(url)
                r.raise_for_status()  # Raise an error for bad status codes
                articles = r.html.find('article')

                if len(articles) > 0:
                    break
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                retries += 1
                print(f"Retrying ({retries}/{max_retries})...")

            # If we exhaust retries, break out of the loop
            if retries == max_retries:
                print(f"Failed to fetch data for {str(current_date.date())} after {max_retries} retries.")
                break

            time.sleep(5)  # Sleep for 5 seconds before retrying

        if retries < max_retries:  # Continue processing if we successfully fetched articles
            print(f"{str(current_date.date())} > {len(articles)} articles fetched")
            allData[str(current_date.date())] = []

            for article in articles:
                t = pq(article.html)
                headingText = t('h2.story__title a.story__link').text()
                spanId = t('span').eq(0).attr('id')
                label = spanId.lower() if spanId is not None else None

                # Only process articles in the "business" or "economy" section
                if len(headingText) > 0 and label in ["business", "economy"]:
                    allData[str(current_date.date())].append({
                        "heading": headingText,
                        "label": label,
                    })
        else:
            print(f"No articles found for {str(current_date.date())}, skipping...")
        
        current_date += dt  # Move to the next day

    # Save the scraped data to an output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(allData, f, ensure_ascii=False, indent=2)

start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 4, 3)  # Corrected second assignment of end_date

output_file = 'C:\\Major Project\\headlines.json'

scrape_news(start_date, end_date, output_file)